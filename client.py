import numpy as np
import torch
import torch.nn.functional as F
from general_utils import accuracy_softmax, accuracy_topk
from loss import MCRLoss, MCRLoss_Basis, CELoss_Basis
import sys
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FF

class MCRClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.dimz = dimz
        self.device = device
        self.neig = neig
        self.num_neig = num_neig
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.mcr_Basisloss = MCRLoss_Basis(eps=0.5, numclasses=self.num_classes)

    def client_plotdata(self, max_images=8, denorm_mean=(0.5,0.5,0.5), denorm_std=(0.5,0.5,0.5)):
        # get one batch
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=8, shuffle=True)
        imgs, labels = next(iter(dataloader))  # imgs: (B, C, H, W)
        imgs = imgs.cpu().detach()

        # limit number of images to plot
        imgs = imgs[:max_images]
        B = imgs.shape[0]

        # debug prints: shape and value ranges
        print(f"[debug] imgs.shape = {imgs.shape}, dtype={imgs.dtype}")
        print(f"[debug] min, max before denorm = {imgs.min().item():.4f}, {imgs.max().item():.4f}")

        # denormalize: x = x * std + mean
        mean = torch.tensor(denorm_mean).view(1, 3, 1, 1)
        std = torch.tensor(denorm_std).view(1, 3, 1, 1)
        imgs = imgs * std + mean

        # clamp to [0,1]
        imgs = imgs.clamp(0.0, 1.0)
        print(f"[debug] min, max after denorm = {imgs.min().item():.4f}, {imgs.max().item():.4f}")

        # convert to numpy HWC for matplotlib
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # (B, H, W, C), floats in [0,1]

        # optional: if colors look swapped, you can try BGR->RGB swap:
        # imgs_np = imgs_np[..., ::-1]

        # plotting
        fig, axs = plt.subplots(1, B, figsize=(B * 2, 2))
        if B == 1:
            axs = [axs]
        for i, im in enumerate(imgs_np):
            axs[i].imshow(im, interpolation='nearest')   # nearest 保持像素格
            axs[i].axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{self.id}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[info] saved {self.id}.png")


    def client_train(self, U_fuse):
        """Update local model using local dataset."""
        # Restore models
        self.netD.train()
        self.netC.eval()
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = 1000, shuffle=True)
        self.netD.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            
            # print(real_cpu.shape)
            real_label = label.detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            err, item1, item2, item3 = self.mcr_Basisloss(Z, real_label, U_fuse)
            loss = err.item()
            loss_item1 = item1.item()
            loss_item2 = item2.item()
            loss_item3 = item3.item()

            err.backward()
            self.optD.step()

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_item1, loss_item2, loss_item3
    

    def client_test(self, U_fuse):
        """Update local model using local dataset."""
        # Restore models
        self.netD.eval()
        self.netC.eval()

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1000, shuffle=True)
            self.netD.to(self.device)

            # while step < self.num_steps:
            for idx, (data, label) in enumerate(dataloader):
                # data, label = next(iter_dataloader)
                # Format batch and label
                real_cpu = data.to(self.device)
                
                # print(real_cpu.shape)
                real_label = label.detach().to(self.device)

                # Forward pass real batch through D
                Z = self.netD(real_cpu)
                err, item1, item2, item3 = self.mcr_Basisloss(Z, real_label, U_fuse)
                loss = err.item()
                loss_item1 = item1.item()
                loss_item2 = item2.item()
                loss_item3 = item3.item()

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_item1, loss_item2, loss_item3

    def getz_train(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = int(len(self.dataset_train)/2), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all

    def getz_all_list(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all, label_all = [], []

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                self.dataset_train, 
                batch_size = int(len(self.dataset_train)/2),  
                shuffle=False
            )
            for data, label in dataloader:
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)

                Z_all.append(Z.cpu().numpy())
                label_all.append(label.cpu().numpy())

        Z_all = np.concatenate(Z_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        return Z_all, label_all

    
    def get_localBasis(self, r_per_class):
        U_local = [None] * self.num_classes  # 预分配列表
        Z_all, label_all = self.getz_all_list()

        # 按类别聚合索引
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, label in enumerate(label_all):
            class_indices[label].append(idx)
        
        # 对每个类别计算局部基
        for c in range(self.num_classes):
            indices = class_indices[c]
            Z_class = np.stack([Z_all[i] for i in indices], axis=0)  # n_i x d
            Z_class_T = Z_class.T  # d x n_i
            r_use = max(1, min(r_per_class, Z_class_T.shape[1]))
            U, S, Vt = np.linalg.svd(Z_class_T, full_matrices=False)
            U_local[c] = U[:, :r_use]  # d x r
        
        return U_local

    def getz_test(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = int(len(self.dataset_test)), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        img_ids_all = np.arange(len(self.dataset_test))           # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all # , img_ids_all


    def client_savemodel(self, path):
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")

class IndepMCRClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.dimz = dimz
        self.device = device
        self.neig = neig
        self.num_neig = num_neig
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.mcr_loss = MCRLoss(eps=0.5, numclasses=self.num_classes)

    def client_plotdata(self, max_images=8, denorm_mean=(0.5,0.5,0.5), denorm_std=(0.5,0.5,0.5)):
        # get one batch
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=8, shuffle=True)
        imgs, labels = next(iter(dataloader))  # imgs: (B, C, H, W)
        imgs = imgs.cpu().detach()

        # limit number of images to plot
        imgs = imgs[:max_images]
        B = imgs.shape[0]

        # debug prints: shape and value ranges
        print(f"[debug] imgs.shape = {imgs.shape}, dtype={imgs.dtype}")
        print(f"[debug] min, max before denorm = {imgs.min().item():.4f}, {imgs.max().item():.4f}")

        # denormalize: x = x * std + mean
        mean = torch.tensor(denorm_mean).view(1, 3, 1, 1)
        std = torch.tensor(denorm_std).view(1, 3, 1, 1)
        imgs = imgs * std + mean

        # clamp to [0,1]
        imgs = imgs.clamp(0.0, 1.0)
        print(f"[debug] min, max after denorm = {imgs.min().item():.4f}, {imgs.max().item():.4f}")

        # convert to numpy HWC for matplotlib
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # (B, H, W, C), floats in [0,1]

        # optional: if colors look swapped, you can try BGR->RGB swap:
        # imgs_np = imgs_np[..., ::-1]

        # plotting
        fig, axs = plt.subplots(1, B, figsize=(B * 2, 2))
        if B == 1:
            axs = [axs]
        for i, im in enumerate(imgs_np):
            axs[i].imshow(im, interpolation='nearest')   # nearest 保持像素格
            axs[i].axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{self.id}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[info] saved {self.id}.png")


    def client_train(self):
        """Update local model using local dataset."""
        # Restore models
        self.netD.train()
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = 1000, shuffle=True)
        self.netD.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            
            # print(real_cpu.shape)
            real_label = label.detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            err, item1, item2 = self.mcr_loss(Z, real_label)
            loss = err.item()
            loss_item1 = item1.item()
            loss_item2 = item2.item()

            err.backward()
            self.optD.step()

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_item1, loss_item2
    

    def client_test(self):
        """Update local model using local dataset."""
        # Restore models
        self.netD.eval()
        dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = min(len(self.dataset_test), 1000), shuffle=True)
        self.netD.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            
            # print(real_cpu.shape)
            real_label = label.detach().to(self.device)

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            err, item1, item2 = self.mcr_loss(Z, real_label)
            loss = err.item()
            loss_item1 = item1.item()
            loss_item2 = item2.item()

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_item1, loss_item2

    def getz_all_list(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all, label_all = [], []

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                self.dataset_train, 
                batch_size = min(int(len(self.dataset_train)/2), 2000),  
                shuffle=False
            )
            for data, label in dataloader:
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)

                Z_all.append(Z.cpu().numpy())
                label_all.append(label.cpu().numpy())

        Z_all = np.concatenate(Z_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        return Z_all, label_all

    def getz_train(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = min(int(len(self.dataset_train)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all

    def getz_test(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = min(int(len(self.dataset_test)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        img_ids_all = np.arange(len(self.dataset_test))           # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all #, img_ids_all


    def client_savemodel(self, path):
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")

class CEClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.device = device
        self.neig = neig
        self.dimz = dimz
        self.num_neig = num_neig
        self.neig_z = 0
        self.neig_z_label = 0
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]

    def client_train(self):
        """Update local model using local dataset."""
        self.netD.train()
        self.netC.train()
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        self.netD.to(self.device)
        self.netC.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            err = F.cross_entropy(output, real_label)
            loss = err.item()
            err.backward()
            self.optD.step()
            self.optC.step()

        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = 1.0*correct/len(self.dataset_train)
        return loss, acc

    def client_test(self):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                loss_cross = F.cross_entropy(output.to(self.device), label.to(self.device))
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss_cross.item(), acc

    def getz_all_list(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all, label_all = [], []

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                self.dataset_train, 
                batch_size = int(len(self.dataset_train)/2),  
                shuffle=False
            )
            for data, label in dataloader:
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)

                Z_all.append(Z.cpu().numpy())
                label_all.append(label.cpu().numpy())

        Z_all = np.concatenate(Z_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        return Z_all, label_all

    
    def get_localBasis(self, r_per_class):
        U_local = [None] * self.num_classes  # 预分配列表
        Z_all, label_all = self.getz_all_list()

        # 按类别聚合索引
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, label in enumerate(label_all):
            class_indices[label].append(idx)
        
        # 对每个类别计算局部基
        for c in range(self.num_classes):
            indices = class_indices[c]
            Z_class = np.stack([Z_all[i] for i in indices], axis=0)  # n_i x d
            Z_class_T = Z_class.T  # d x n_i
            r_use = max(1, min(r_per_class, Z_class_T.shape[1]))
            U, S, Vt = np.linalg.svd(Z_class_T, full_matrices=False)
            U_local[c] = U[:, :r_use]  # d x r
        
        return U_local
    
    def getz_train(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = min(int(len(self.dataset_train)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all

    def getz_test(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = min(int(len(self.dataset_test)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        img_ids_all = np.arange(len(self.dataset_test))           # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all # , img_ids_all

    def client_savemodel(self, path):
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")


class svdCEClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.device = device
        self.neig = neig
        self.dimz = dimz
        self.num_neig = num_neig
        self.neig_z = 0
        self.neig_z_label = 0
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.loss = CELoss_Basis(numclasses=self.num_classes)

    def client_train(self, U_fuse):
        """Update local model using local dataset."""
        self.netD.train()
        self.netC.train()
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        self.netD.to(self.device)
        self.netC.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            err, item1, item2 = self.loss(Z, output, real_label, U_fuse)
            loss = err.item()
            err.backward()
            self.optD.step()
            self.optC.step()

        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = 1.0*correct/len(self.dataset_train)
        return loss, item1.item(), item2.item(), acc

    def client_test(self, U_fuse):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                err, item1, item2 = self.loss(feature, output, label.to(self.device), U_fuse)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)

        if self.device == "cuda": torch.cuda.empty_cache()
        return err.item(), acc

    def getz_all_list(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all, label_all = [], []

        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(
                self.dataset_train, 
                batch_size = int(len(self.dataset_train)/2),  
                shuffle=False
            )
            for data, label in dataloader:
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)

                Z_all.append(Z.cpu().numpy())
                label_all.append(label.cpu().numpy())

        Z_all = np.concatenate(Z_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        return Z_all, label_all

    
    def get_localBasis(self, r_per_class):
        U_local = [None] * self.num_classes  # 预分配列表
        Z_all, label_all = self.getz_all_list()

        # 按类别聚合索引
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, label in enumerate(label_all):
            class_indices[label].append(idx)
        
        # 对每个类别计算局部基
        for c in range(self.num_classes):
            indices = class_indices[c]
            Z_class = np.stack([Z_all[i] for i in indices], axis=0)  # n_i x d
            Z_class_T = Z_class.T  # d x n_i
            r_use = max(1, min(r_per_class, Z_class_T.shape[1]))
            U, S, Vt = np.linalg.svd(Z_class_T, full_matrices=False)
            U_local[c] = U[:, :r_use]  # d x r
        
        return U_local
    
    def getz_train(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = min(int(len(self.dataset_train)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all

    def getz_test(self):
        self.netD.to(self.device).eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        img_ids_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_test, batch_size = min(int(len(self.dataset_test)/2), 2000), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD(real_cpu)  # [batch_size, D]
            
                Z_all.append(Z.cpu())           # 保存特征
                label_all.append(label.cpu())   # 保存标签
                
        Z_all = np.concatenate(Z_all, axis=0)          # [Num, D]
        label_all = np.concatenate(label_all, axis=0).squeeze()  # [Num]
        img_ids_all = np.arange(len(self.dataset_test))           # [Num]
        del dataloader, data, real_cpu
        return Z_all, label_all #, img_ids_all

    def client_savemodel(self, path):
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")