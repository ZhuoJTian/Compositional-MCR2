from __future__ import absolute_import
from __future__ import print_function

from client import MCRClient
import torch
import copy
import numpy as np

# import datasets_old
from model import Encoder, Classifier
from args import parse_args
from Sample_parti import getAttr
from datasets import cifar10
from general_utils import plot_corZ

def create_clients(datasets_train, datasets_test, num_classes, dim_z, adj, path):
    clients = []
    for k in range(len(datasets_train)):
        neig_location = list(np.nonzero(adj[:, k])[0])
        num_neig = len(neig_location)
        client = MCRClient(client_id=k,
                        data_set_train=datasets_train[k],
                        data_set_test=datasets_test[k],
                        num_classes=num_classes,
                        dimz=dim_z,
                        neig=neig_location,
                        num_neig=num_neig,
                        device="cuda",
                        path = path)
        clients.append(client)
    return clients

def set_up_clients(args, adj, models):
    _, LocalDist, LocaDis_test, _, _, _ = getAttr(4)
    view_starts = [[0, 0], [14, 0], [0, 14], [14, 14]]
    datasets_train, datasets_test = cifar10.create_datasets(
        args.num_clients, view_starts, 18, LocalDist, LocaDis_test
    )

    # plot_datasets(args.num_clients, local_datasets_train)
    print("load the datasets")
    # Ini_model = RestNet18_att()
    clients = create_clients(datasets_train, datasets_test, 
                             args.num_classes, args.dim_z, adj, args.path_result)
    
    for client_id in range(args.num_clients):
        client = clients[client_id]
        model = models[client_id]
        encoder = Encoder(args.dim_z, model)
        classifier= Classifier(args.num_classes, args.dim_z)
         # initialize model parameters
        
        client.netD = copy.deepcopy(encoder).to("cpu")
        path_model = "./CIFAR/Exp_results2/ColMCR/"+"models/"+ str(client_id)
        checkpoint = torch.load(path_model + "_netD.pt")
        client.netD = copy.deepcopy(encoder).to("cpu")
        client.netD.load_state_dict(checkpoint['model_state_dict'])
        client.netD.eval()
        opt_D = torch.optim.Adam(client.netD.parameters(), lr=0.01, weight_decay=args.weight_decay)
        opt_D.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in opt_D.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        client.netC = copy.deepcopy(classifier).to("cpu")
        opt_C = torch.optim.Adam(client.netC.parameters(), lr=0.01, weight_decay=args.weight_decay)
        del(encoder, classifier)
        
        '''
        client.netD = copy.deepcopy(encoder).to("cpu")
        client.netC = copy.deepcopy(classifier).to("cpu")
        del(encoder, classifier)
        opt_D = torch.optim.Adam(client.netD.parameters(), lr=0.01, weight_decay=args.weight_decay)
        opt_C = torch.optim.Adam(client.netC.parameters(), lr=0.01, weight_decay=args.weight_decay)
        '''
        client.setup(batch_size=args.batch_size,
                     num_local_epochs=1,
                     optimizer_D=opt_D,
                     optimizer_C=opt_C
                     )
        del opt_D, opt_C
    print("load the clients and initialized the parameters")
    return clients

def pca_svd(X, r):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :r]

def randomized_svd(X, r, n_iter, seed=0):
    rng = np.random.default_rng(seed)
    d, n = X.shape
    Omega = rng.standard_normal((n, r))
    Y = X @ Omega
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ X
    Ub, Sb, Vtb = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :r]


def refresh_fusion_basis(clients, r=10, R=16, num_classes=10, use_randomized=False, n_iter=2, seed=0):
    # 1. 获取每个客户端的局部基
    U_local_all = [None] * len(clients)
    for i, client in enumerate(clients):
        U_local_all[i] = client.get_localBasis(r)
        
    # 2. 融合每个类别的基
    U_fuse = [None] * num_classes
    for k in range(num_classes):
        # 拼接所有客户端同类的局部基
        U_cat = np.concatenate([U_local_all[i][k] for i in range(len(clients))], axis=1)  # d x sum r_i
        
        # 随机化 SVD 或标准 SVD
        if use_randomized:
            U_fuse[k] = randomized_svd(U_cat, r=R, n_iter=n_iter, seed=seed + 999)
        else:
            U, S, Vt = np.linalg.svd(U_cat, full_matrices=False)
            U_fuse[k] = torch.tensor(U[:, :R]).float()

    del U_local_all
    return U_fuse


def main():
    args = parse_args()
    args.num_clients = 4
    adj = np.ones((args.num_clients, args.num_clients)) - np.eye(args.num_clients)

    args.num_classes = 10
    args.path_result = "./CIFAR/Exp_results2/ColMCR2/"
    args.epochs = 8000
    models = ['res18', 'res34', 'vgg11', 'vgg16']

    clients = set_up_clients(args, adj, models)
    print("client set up")
    path_model = args.path_result+"models/"
    # Start training

    U_fuse = refresh_fusion_basis(clients)

    for epoch in range(4000, args.epochs):
        train_loss_all = []
        test_loss_all = []
        Z_all = []
        label_all = []
        agent_all = []
        for client_id in range(args.num_clients): # tqdm(range(args.num_clients), ascii=True):  
            client = clients[client_id]
            train_loss, loss_term1, loss_term2, loss_term3 = client.client_train(U_fuse)
            test_loss, _, _, _ = client.client_test(U_fuse)
            train_loss_all.append(train_loss)
            test_loss_all.append(test_loss)

            if (epoch+1)%1==0:
                # client.client_savemodel(path_model)
                file_path = args.path_result + "loss_" + str(client_id) + ".txt"
                with open(file_path, "a+") as f:
                    f.write("epoch {} ".format(epoch+1))
                    f.write("Trainloss: {:.4f}, Testloss: {:.4f}, Trainloss_term1: {:.4f}, Trainloss_term2: {:.4f}, Trainloss_term3: {:.4f}".format(\
                        train_loss, test_loss, loss_term1, loss_term2, loss_term3))
                    f.write("\n")
        
            if (epoch+1)%200==0:
                client.client_savemodel(path_model)
                Z_list, label_list = client.getz_all_list()
                if len(Z_all) == 0:
                    Z_all = Z_list
                    label_all = label_list
                    agent_all = np.array([client_id] * len(Z_list))
                else:
                    Z_all = np.concatenate([Z_all, Z_list], axis=0)
                    label_all = np.concatenate([label_all, label_list], axis=0)
                    agent_all = np.concatenate([agent_all, np.array([client_id] * len(Z_list))], axis=0)
                del Z_list, label_list

        U_fuse = refresh_fusion_basis(clients)

        if (epoch+1)%1==0:
            file_path = args.path_result + "averaged_result" + ".txt"
            with open(file_path, "a+") as f:
                f.write("epoch {} ".format(epoch+1))
                f.write("Trainloss: {:.4f}, Testloss: {:.4f}".format(np.average(train_loss_all), np.average(test_loss_all)))
                f.write("\n")

        if (epoch+1)%200 == 0:
            path_fig = args.path_result + "CorZ_" + str(epoch+1) + ".jpg"
            plot_corZ(Z_all, label_all, agent_all, path_fig)

        # evaluate on test set
    print("end of training")


if __name__ == "__main__":
    main()
