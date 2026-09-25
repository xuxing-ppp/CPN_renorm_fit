# import torch
# import torch.nn as nn
import copy
import scipy
import numpy as np


# def from_phi_to_z(phi):
#     '''
#     From a (bs, 2N)-dim real tensor to a (bs, N)-dim complex tensor, pytorch
#     '''
#     phi = torch.as_tensor(phi)
#     phi0 = phi.reshape(phi.shape[0],-1,2)
#     return phi0[:,:,0] + 1j*phi0[:,:,1]

# def z_vdot(z1, z2):
#     '''
#     z1, z2: (bs, N)-dim tensor, calculate conj(z1)z2 and extract norm, cos, sin, return (bs, 3)-dim tensor
#     '''
#     z1 = torch.as_tensor(z1)
#     z2 = torch.as_tensor(z2)
#     temp = torch.sum(torch.conj(z1) * z2, dim=1)
#     # temp = torch.einsum("ij,ij->i", torch.conj(z1), z2)
#     ang_temp = torch.angle(temp)
#     return torch.stack([torch.abs(temp), torch.cos(ang_temp), torch.sin(ang_temp)], dim=1).float()

# def z_vdot_angle(z1, z2):
#     '''
#     z1, z2: (bs, N)-dim tensor, calculate conj(z1)z2 and extract norm, arg, return (bs, 2)-dim tensor
#     '''
#     z1 = torch.as_tensor(z1)
#     z2 = torch.as_tensor(z2)
#     temp = torch.sum(torch.conj(z1) * z2, dim=1)
#     # temp = torch.einsum("ij,ij->i", torch.conj(z1), z2)
#     ang_temp = torch.angle(temp)
#     return torch.stack([torch.abs(temp), ang_temp], dim=1).float()

# -----------------------------------------Fitting Actions-------------------------------------
# using numpy
def z1z4_extract(phi):
    '''
    :param phi: (bs, 2 * 6 * N)-dim tensor
    :return z1z4: (bs, 2) tensor, z1z4 inner product, norm and angle
    '''
    def from_phi_to_z_np(phi):
        phi = np.asarray(phi)
        phi0 = phi.reshape(phi.shape[0],-1,2)
        return phi0[:,:,0] + 1j*phi0[:,:,1]
    N = (phi.shape[1]) // 12
    z1 = from_phi_to_z_np(phi[:, 1*2*N:2*2*N])
    z4 = from_phi_to_z_np(phi[:, 4*2*N:5*2*N])
    temp = np.sum(np.conj(z1) * z4, axis=1)
    ang_temp = np.angle(temp)
    return np.stack([np.abs(temp), ang_temp], axis=1)

class Villain_fit():
    def __init__(self, N):
        self.N = N

    def Villain_dist_norm_fit(self, X, beta, alpha):
        '''
        :param X: (bs, 1 + 6 + 2)-dim array, a, a_array, z1z4
        '''
        U = np.exp(1j * X[:,0])
        U_list = np.exp(1j * X[:, 1:7])
        z1z4 = X[:, 7:]

        F = 2 * self.N * beta * z1z4[:, 0] * np.exp(-1j * z1z4[:, 1]) + alpha * U_list[:, 0] * U_list[:, 4] * U_list[:, 5] + alpha * np.conj(U_list[:, 1] * U_list[:, 2] * U_list[:, 3])
        return np.exp(np.real(U * F)) / scipy.special.i0(np.abs(F)) / 2 / np.pi

# # Villain action
# def Villain_action_func(a, beta, alpha, params):
#     U_list = np.exp(1j * params[:6])
#     U = np.exp(1j * a)
#     N = (len(params) - 6) // 12
#     phi1, phi4 = params[6+2*N:6+4*N], params[6+8*N:6+10*N]
#     z1, z4 = from_phi_to_z(phi1), from_phi_to_z(phi4)

#     F = 2 * N * beta * np.vdot(z4, z1) + alpha * U_list[0] * U_list[4] * U_list[5] + alpha * np.conj(U_list[1] * U_list[2] * U_list[3])
#     return -np.real(U * F)

# # Plaquette dist
# def Villain_plaq_dist_norm(a, alpha, params):
#     U_list = np.exp(1j * params[:6])
#     U = np.exp(1j * a)

#     F = alpha * U_list[0] * U_list[4] * U_list[5] + alpha * np.conj(U_list[1] * U_list[2] * U_list[3])
#     return np.exp(np.real(U * F)) / special.i0(np.abs(F)) / 2 / np.pi

# class aDist_Villain(nn.Module):
#     '''
#     Villian a-distribution, with beta and alpha as trainable parameters.
#     '''
#     def __init__(self, N, beta0=0, alpha0=0, device="cuda"):
#         super().__init__()
#         self.N = N
#         self.device = device
#         self.beta = nn.Parameter(torch.tensor(beta0, dtype=torch.float32, device=device))
#         self.alpha = nn.Parameter(torch.tensor(alpha0, dtype=torch.float32, device=device))

#     def X_extract(self, X):
#         '''
#         :return a: (bs,) tensor
#         :return a_tensor: (bs, 6) tensor
#         :return z1z4: (bs, 2) tensor, z1z4 inner product, norm and angle
#         '''
#         N = (X.shape[1] - 7) // 12
#         a = X[:,0]
#         a_tensor = X[:, 1:7]
#         z1 = from_phi_to_z(X[:, 7+1*2*N:7+2*2*N])
#         z4 = from_phi_to_z(X[:, 7+4*2*N:7+5*2*N])
#         z1z4 = z_vdot_angle(z1, z4)
#         return a, a_tensor, z1z4
    
#     def z1z4_extract(self, phi):
#         '''
#         :param phi: (bs, 2 * 6 * N)-dim tensor
#         :return z1z4: (bs, 2) tensor, z1z4 inner product, norm and angle
#         '''
#         N = (phi.shape[1]) // 12
#         z1 = from_phi_to_z(phi[:, 1*2*N:2*2*N])
#         z4 = from_phi_to_z(phi[:, 4*2*N:5*2*N])
#         z1z4 = z_vdot_angle(z1, z4)
#         return z1z4

#     def forward(self, a, a_tensor, z1z4):
#         U = torch.exp(1j * a)  # (bs,)
#         U_list = torch.exp(1j * a_tensor)  # (bs, 6)

#         F = 2 * self.N * self.beta * z1z4[:, 0] * torch.exp(-1j * z1z4[:, 1]) + self.alpha * U_list[:,0] * U_list[:,4] * U_list[:,5] + self.alpha * torch.conj(U_list[:,1] * U_list[:,2] * U_list[:,3])
#         out = torch.exp(torch.real(U * F)) / torch.special.i0(torch.abs(F)) / 2 / torch.pi
#         return out.unsqueeze(1)

#     def dist_X(self, X):
#         '''
#         :param X: (bs, 7 + 6 * 2 * N) tensor
#         '''
#         U = torch.exp(1j * X[:,0])  # (bs,)
#         U_list = torch.exp(1j * X[:, 1:7])  # (bs, 6)
#         N = (X.size(1) - 7) // 12
#         phi1, phi4 = X[:, 7+2*N:7+4*N], X[:, 7+8*N:7+10*N]
#         z1, z4 = from_phi_to_z(phi1), from_phi_to_z(phi4)

#         # z4z1 = torch.einsum("ij,ij->i", torch.conj(z4), z1)
#         z4z1 = torch.sum(torch.conj(z4) * z1, dim=1)
#         F = 2 * N * self.beta * z4z1 + self.alpha * U_list[:,0] * U_list[:,4] * U_list[:,5] + self.alpha * torch.conj(U_list[:,1] * U_list[:,2] * U_list[:,3])
#         out = torch.exp(torch.real(U * F)) / torch.special.i0(torch.abs(F)) / 2 / torch.pi
#         return out.unsqueeze(1)

# if __name__ == "__main__":
#     model3 = aDist_Villain(N=4, beta0=0.5, alpha0=0.5, device="mps")
#     print(sum(p.numel() for p in model3.parameters()))