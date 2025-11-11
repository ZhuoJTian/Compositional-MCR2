import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MCRLoss(nn.Module):

    def __init__(self, eps, numclasses):
        super(MCRLoss, self).__init__()

        self.num_class = numclasses
        self.faster_logdet = False
        self.eps = eps

    def forward(self, Z, real_label):
        err, item1, item2 = self.fast_version(Z, real_label)
        return err, item1, item2

    def fast_version(self, Z, real_label):
        """ decrease the times of calculate log-det  from 52 to 32"""
        z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label,
                                                                                                self.num_class)
        errD_mcr = z_discrimn_item - z_compress_item
        errD = -1.0 * errD_mcr 
        return errD, -1.0 * z_discrimn_item, z_compress_item

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)


class MCRLoss_Basis(nn.Module):

    def __init__(self, eps, numclasses):
        super(MCRLoss_Basis, self).__init__()

        self.num_class = numclasses
        self.faster_logdet = False
        self.eps = eps
        self.lam = 100.0

    def forward(self, Z, real_label, U_fuse):

        """ decrease the times of calculate log-det  from 52 to 32"""
        z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label,
                                                                                                self.num_class)
        errD_mcr = z_discrimn_item - z_compress_item
        errD = -1.0 * errD_mcr
        err_rec = self.compute_basis(Z, real_label, U_fuse)

        err = errD + self.lam*err_rec

        return err, -1.0 * z_discrimn_item, z_compress_item, self.lam*err_rec

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)
    
    def compute_basis(self, Z, Y, U_fuse):
        """
        Z: [N, d]
        Y: [N]  标签
        U_fuse: list, each element: [d, k]
        """
        # one-hot 编码 [N, C]
        # 假设 U_fuse 是列表，每个 U_fuse[j] 都是 [d, r]
        U_fuse_tensor = torch.stack(U_fuse, dim=0).to(Z.device)  # [num_classes, d, r]

        # 确保 Y 在同一设备
        Y = Y.to(Z.device)

        # 批量选择每个样本对应的基
        U_fuse_expanded = U_fuse_tensor[Y]  # [N, d, r]

        # 投影
        proj = torch.bmm(torch.bmm(Z.unsqueeze(1), U_fuse_expanded), U_fuse_expanded.transpose(1, 2))
        proj = proj.squeeze(1)  # [N, d]

        # MSE
        err_rec = F.mse_loss(proj, Z)
        return err_rec


class CELoss_Basis(nn.Module):

    def __init__(self, numclasses):
        super(CELoss_Basis, self).__init__()

        self.num_class = numclasses
        self.lam = 10.0

    def forward(self, Z, output, real_label, U_fuse):

        """ decrease the times of calculate log-det  from 52 to 32"""
        errD = F.cross_entropy(output, real_label)

        err_rec = self.compute_basis(Z, real_label, U_fuse)

        err = errD + self.lam*err_rec

        return err, errD, self.lam*err_rec

    def compute_basis(self, Z, Y, U_fuse):
        """
        Z: [N, d]
        Y: [N]  标签
        U_fuse: list, each element: [d, k]
        """
        # one-hot 编码 [N, C]
        # 假设 U_fuse 是列表，每个 U_fuse[j] 都是 [d, r]
        U_fuse_tensor = torch.stack(U_fuse, dim=0).to(Z.device)  # [num_classes, d, r]

        # 确保 Y 在同一设备
        Y = Y.to(Z.device)

        # 批量选择每个样本对应的基
        U_fuse_expanded = U_fuse_tensor[Y]  # [N, d, r]

        # 投影
        proj = torch.bmm(torch.bmm(Z.unsqueeze(1), U_fuse_expanded), U_fuse_expanded.transpose(1, 2))
        proj = proj.squeeze(1)  # [N, d]

        # MSE
        err_rec = F.mse_loss(proj, Z)
        return err_rec


