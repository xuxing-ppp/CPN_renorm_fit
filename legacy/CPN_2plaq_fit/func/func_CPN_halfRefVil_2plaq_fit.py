import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import scipy
from numpy.random import rand, randn, vonmises
from numpy.fft import fftn, ifftn
from func.func_CPN_halfRefVil_HMC import CPN_halfRefVil_HMCSampler, CPN_halfRefVil_HMCSampler_OBC

def normalize_z(z):
    z = np.array(z, dtype=complex)
    norm = np.linalg.norm(z)
    if norm < 1e-8:
        raise Exception("zero norm")
    return z / norm

def from_z_to_phi(z):
    '''
    From a k-dim complex vector to a 2k-dim real vector
    '''
    l = [[np.real(zi), np.imag(zi)] for zi in z]
    return np.concatenate(l)

def from_phi_to_z(phi):
    '''
    From a 2k-dim real vector to a k-dim complex vector
    '''
    phi0 = phi.reshape(-1,2)
    return phi0[:,0] + 1j*phi0[:,1]

def vdot_z(z1, z2):
    return np.einsum('ijk,ijk->ij', np.conj(z1), z2)

def autocorr_fft(x):
    """
    Autocorrelation of a (possibly complex) 1D array using full FFT.
    Returns ACF for lags 0..N-1, normalized so that ACF[0] = 1.
    """
    x = np.asarray(x)
    n = x.size
    x = x - x.mean()
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n]
    acf = acf.real
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time for real or complex arrays.
    If x is multidimensional, compute along the last axis.
    """
    x = np.asarray(x)
    if x.ndim > 1:
        return np.apply_along_axis(integrated_autocorr_time, -1, x, c)
    ac = autocorr_fft(x)
    tau = 0.5
    for t in range(1, len(ac)):
        tau += ac[t]
        if t > c * tau:
            break
    return tau

def U1_clip(x):
    return np.mod(x + np.pi, 2*np.pi) - np.pi

class CPN_halfRefVil_Sampler_2plaq:
    """
    CP^(N-1) Over-Heatbath Monte-Carlo sampler, OBC with fixed boundary.

    System size: (2L+1) * (L+1) (2plaq)

    N-dim normalized complex vector on sites, physical U(1) on links,
    and auxiliary U(1) link field U1.

    Refined half-Villainized action:
      2N * (beta * sum (1 - Re(z' U z)) + beta1 * sum (1 - Re(z' U1 z))) + alpha * sum (1 - Re(UUUU))

    Boundary is fixed by given z_list and U_list.

    :param beta:  coupling between z and physical U
    :param beta1: coupling between z and auxiliary U1 (z-z interaction)
    :param alpha: plaquette coupling
    """

    def __init__(self, N, L, beta, beta1, alpha, z_list, U_list, seed=None):
        self.L = L
        self.Lx = 2*L + 1
        self.Ly = L + 1
        if N <= 1:
            raise ValueError("N too small")
        self.N = N
        self.beta = beta
        self.beta1 = beta1
        self.alpha = alpha
        if not(len(z_list) == 6*L and len(U_list) == 6*L):
            raise IndexError("length of z_list or U_list incorrect")
        self.z_list = z_list
        self.U_list = U_list

        if seed is not None:
            np.random.seed(seed)

        self.z = self._random_unit_vectors((self.Lx, self.Ly, self.N))
        phases = 2*np.pi * rand(self.Lx, self.Ly, 2)
        self.U = np.exp(1j * phases)
        phases = 2*np.pi * rand(self.Lx, self.Ly, 2)
        self.U1 = np.exp(1j * phases)

        self.boundary_mask_z = np.zeros((self.Lx, self.Ly), dtype=bool)
        self.boundary_mask_U = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self.boundary_mask_U1 = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self._init_config()

    def _init_config(self):
        for x in range(self.Lx):
            for y in range(self.Ly):
                if x == 0 or x == self.Lx - 1 or y == 0 or y == self.Ly - 1:
                    self.boundary_mask_z[x, y] = True
                    self.z[x, y] = normalize_z(self._boundary_z(x, y, self.z_list))
        for x in range(self.Lx):
            for y in (0, self.Ly - 1):
                self.boundary_mask_U[x, y, 0] = True
                self.boundary_mask_U1[x, y, 0] = True
                self.U[x, y, 0] = normalize_z(self._boundary_U(x, y, 0, self.U_list))
        for y in range(self.Ly):
            for x in (0, self.Lx - 1):
                self.boundary_mask_U[x, y, 1] = True
                self.boundary_mask_U1[x, y, 1] = True
                self.U[x, y, 1] = normalize_z(self._boundary_U(x, y, 1, self.U_list))

    # ------------------------------------------------------------
    # -------------------------- Utility ---------------------
    # ------------------------------------------------------------

    def _random_unit_vectors(self, shape):
        """
        Samples random complex unit vectors uniformly on S^{2N-1}.
        shape = (Lx,Ly,N)
        """
        real = randn(*shape)
        imag = randn(*shape)
        v = real + 1j * imag
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm

    def _boundary_z(self, x, y, z_list):
        '''
        Generate the boundary z configuration according to z_list.
        '''
        L = self.L
        if y == 0:
            return z_list[x]
        elif x == 2 * L:
            return z_list[y + 2*L]
        elif y == L:
            return z_list[5*L - x]
        elif x == 0:
            return z_list[6*L - y]
        else:
            return self._random_unit_vectors(self.N)
        
    def _boundary_U(self, x, y, mu, U_list):
        '''
        Generate the boundary U configuration according to U_list.
        '''
        L = self.L
        if y == 0 and mu == 0:
            return U_list[x]
        elif x == 2 * L and mu == 1:
            return U_list[2*L + y]
        elif y == L and mu == 0:
            return np.conj(U_list[5*L - 1 - x])
        elif x == 0 and mu == 1:
            return np.conj(U_list[6*L - 1 - y])
        else:
            return np.exp(1j * 2 * np.pi * rand())

    # ------------------------------------------------------------
    # ---------- Force terms --------------------------------------
    # ------------------------------------------------------------

    def force_z(self, x, y):
        """
        Compute local force at site (x, y) with open boundary conditions.
        Includes contributions from both physical U and auxiliary U1.
        Returns shape (N,) complex.
        """
        if x == 0 or x == self.Lx - 1 or y == 0 or y == self.Ly - 1:
            raise Exception("boundary update")

        Lx, Ly = self.Lx, self.Ly
        z = self.z
        U = self.U
        U1 = self.U1
        N = self.N

        F = np.zeros(N, dtype=complex)
        F1 = np.zeros(N, dtype=complex)

        if x > 0:
            F += U[x - 1, y, 0] * z[x - 1, y]
            F1 += U1[x - 1, y, 0] * z[x - 1, y]
        if x < Lx - 1:
            F += np.conj(U[x, y, 0]) * z[x + 1, y]
            F1 += np.conj(U1[x, y, 0]) * z[x + 1, y]
        if y > 0:
            F += U[x, y - 1, 1] * z[x, y - 1]
            F1 += U1[x, y - 1, 1] * z[x, y - 1]
        if y < Ly - 1:
            F += np.conj(U[x, y, 1]) * z[x, y + 1]
            F1 += np.conj(U1[x, y, 1]) * z[x, y + 1]

        return 2 * N * self.beta * F + 2 * N * self.beta1 * F1

    def force_U(self, x, y, mu):
        """
        Force for physical U(1) link:
        F = 2N*beta z(x+mu) dot conj(z(x)) + alpha * (plaq terms)
        return a complex number
        """
        if (x == 0 and mu == 1) or x == self.Lx - 1 or (y == 0 and mu == 0) or y == self.Ly - 1:
            raise Exception("boundary update")

        xp = (x + 1)
        xm = (x - 1)
        yp = (y + 1)
        ym = (y - 1)

        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        alpha = self.alpha

        if mu == 0:
            return 2 * N * beta * np.vdot(z[x, y], z[xp, y]) + alpha * (np.conj(U[xp, y, 1]) * U[x, yp, 0] * U[x, y, 1] + U[xp, ym, 1] * U[x, ym, 0] * np.conj(U[x, ym, 1]))
        elif mu == 1:
            return 2 * N * beta * np.vdot(z[x, y], z[x, yp]) + alpha * (np.conj(U[x, yp, 0]) * U[xp, y, 1] * U[x, y, 0] + U[xm, yp, 0] * U[xm, y, 1] * np.conj(U[xm, y, 0]))
        else:
            raise Exception("mu out of range")

    def force_U1(self, x, y, mu):
        """
        Force for auxiliary U(1) link:
        F = 2N*beta1 z(x+mu) dot conj(z(x))
        return a complex number
        """
        if (x == self.Lx - 1 and mu == 0) or (y == self.Ly - 1 and mu == 1):
            raise Exception("boundary update")

        z = self.z
        N = self.N
        beta1 = self.beta1

        if mu == 0:
            return 2 * N * beta1 * np.vdot(z[x, y], z[x + 1, y])
        elif mu == 1:
            return 2 * N * beta1 * np.vdot(z[x, y], z[x, y + 1])
        else:
            raise Exception("mu out of range")

    # ------------------------------------------------------------
    # ---------- Local Updates (Heatbath) -------------------------
    # ------------------------------------------------------------

    def hb_update_z(self, x, y):
        """
        Proper heatbath update for z(x, y).
        Uses correct CP^{N-1} sampling of angle relative to F.
        """
        F = self.force_z(x, y)
        Fnorm = np.linalg.norm(F)

        if Fnorm < 1e-10:
            self.z[x, y] = self._random_unit_vectors((1, self.N))[0]
            return

        rnorm = 0
        while rnorm < 1e-10:
            r = randn(self.N) + 1j*randn(self.N)
            r -= (np.real(np.vdot(F, r)) / Fnorm**2) * F
            rnorm = np.linalg.norm(r)

        r /= rnorm
        N = self.N

        def p_k(theta):
            return (np.sin(theta)**(2*N-2)) * np.exp(Fnorm * np.cos(theta))

        def theta_0(k, a):
            return np.acos(np.sqrt(1 + ((k-1) / a)**2) - ((k-1) / a))

        def c_0(k, a):
            t0 = theta_0(k, a)
            return np.sqrt(2 * (k-1) * (1 - ((k-1) / a) * np.cos(t0)) / np.sin(t0)**2)

        t0 = theta_0(N, Fnorm)
        c = c_0(N, Fnorm)
        eta = 0.97
        while True:
            chi = rand()
            t_trial = t0 + np.tan(chi * np.atan(c * (np.pi - t0)) + (chi - 1) * np.atan(c * t0)) / c
            p_acc = eta * p_k(t_trial) / p_k(t0) * (1 + c**2 * (t_trial - t0)**2)
            if p_acc > 1:
                raise Exception('Acceptance probability larger than 1')
            if rand() < p_acc:
                break
        theta = t_trial

        new_z = np.cos(theta) * (F / Fnorm) + np.sin(theta) * r
        self.z[x, y] = new_z / np.linalg.norm(new_z)

    def hb_update_U(self, x, y, mu):
        """
        Over-heatbath update for physical U(1) link.
        Sample angle from distribution:
            P(theta) ∝ exp( |F| cos(theta - arg(F)) )
        """
        F = self.force_U(x, y, mu)
        Fmag = np.abs(F)

        if Fmag < 1e-10:
            self.U[x, y, mu] = np.exp(1j * 2*np.pi * rand())
            return

        new_phase = vonmises(np.angle(F), Fmag)
        self.U[x, y, mu] = np.exp(1j * new_phase)

    def hb_update_U1(self, x, y, mu):
        """
        Over-heatbath update for auxiliary U(1) link.
        Sample angle from distribution:
            P(theta) ∝ exp( |F| cos(theta - arg(F)) )
        """
        F = self.force_U1(x, y, mu)
        Fmag = np.abs(F)

        if Fmag < 1e-10:
            self.U1[x, y, mu] = np.exp(1j * 2*np.pi * rand())
            return

        new_phase = vonmises(np.angle(F), Fmag)
        self.U1[x, y, mu] = np.exp(1j * new_phase)

    # ------------------------------------------------------------
    # ---------- Local Updates (Overrelaxation) -------------------
    # ------------------------------------------------------------

    def or_update_z(self, x, y):
        F = self.force_z(x, y)
        Fnorm = np.linalg.norm(F)
        z = self.z[x, y]
        z_new = 2 * np.real(np.vdot(z, F)) / Fnorm**2 * F - z
        self.z[x, y] = z_new / np.linalg.norm(z_new)

    def or_update_U(self, x, y, mu):
        F = self.force_U(x, y, mu)
        Fnorm = np.linalg.norm(F)
        U = self.U[x, y, mu]
        U_new = 2 * np.real(np.conj(U) * F) / Fnorm**2 * F - U
        self.U[x, y, mu] = U_new / np.abs(U_new)

    def or_update_U1(self, x, y, mu):
        F = self.force_U1(x, y, mu)
        Fnorm = np.linalg.norm(F)
        U1 = self.U1[x, y, mu]
        U1_new = 2 * np.real(np.conj(U1) * F) / Fnorm**2 * F - U1
        self.U1[x, y, mu] = U1_new / np.abs(U1_new)

    # ------------------------------------------------------------
    # ---------- MC Sweep -----------------------------------------
    # ------------------------------------------------------------

    def sweep(self, heatbath_fraction=0.4):
        """Heatbath + overrelaxation sweep."""
        # mu = -1: z update;  0, 1: U update;  2, 3: U1 update
        if abs(self.beta1) > 1e-10:
            indices = [(x, y, mu) for y in range(self.Ly-1) for x in range(self.Lx-1) for mu in (-1, 0, 1, 2, 3)]
        else:
            indices = [(x, y, mu) for y in range(self.Ly-1) for x in range(self.Lx-1) for mu in (-1, 0, 1)]
        np.random.shuffle(indices)
        for (x, y, mu) in indices:
            if mu == -1:
                if self.boundary_mask_z[x, y]:
                    continue
                if rand() < heatbath_fraction:
                    self.hb_update_z(x, y)
                else:
                    self.or_update_z(x, y)
            elif mu <= 1:
                if self.boundary_mask_U[x, y, mu]:
                    continue
                if rand() < heatbath_fraction:
                    self.hb_update_U(x, y, mu)
                else:
                    self.or_update_U(x, y, mu)
            else:
                mu1 = mu - 2
                if self.boundary_mask_U1[x, y, mu1]:
                    continue
                if rand() < heatbath_fraction:
                    self.hb_update_U1(x, y, mu1)
                else:
                    self.or_update_U1(x, y, mu1)

    # ------------------------------------------------------------
    # ---------- Measurements -------------------------------------
    # ------------------------------------------------------------

    def energy_density(self):
        """
        Compute (S / V / 2), where
          S = 2 * beta  * sum (1 - Re(z' U z))
            + beta1 * sum (1 - |z' z|^2 / 2)

        V = total number of links.
        Returns float.
        """
        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        beta1 = self.beta1
        Lx, Ly = self.Lx, self.Ly
        Vx = (Lx - 1) * Ly   # number of x-links
        Vy = Lx * (Ly - 1)   # number of y-links
        V = Vx + Vy

        E_beta = 0.0
        E_beta1 = 0.0

        for x in range(Lx - 1):
            for y in range(Ly):
                E_beta += 1 - np.real(U[x, y, 0] * np.vdot(z[x + 1, y], z[x, y]))
                E_beta1 += 0.5 - np.abs(np.vdot(z[x + 1, y], z[x, y]))**2 / 2

        for x in range(Lx):
            for y in range(Ly - 1):
                E_beta += 1 - np.real(U[x, y, 1] * np.vdot(z[x, y + 1], z[x, y]))
                E_beta1 += 0.5 - np.abs(np.vdot(z[x, y + 1], z[x, y]))**2 / 2

        return (beta * E_beta + beta1 * E_beta1) / (beta + beta1) / V

    def wilson_loop11(self):
        '''
        Compute average value of a 1*1 Wilson loop i.e. Re(UUUU) = cos(da).
        Use two methods: Re(UUUU) and cos(arg(ZZ) + arg(ZZ) + ...), and return two float.
        '''
        w_U = 0
        w_z = 0
        U = self.U
        z = self.z
        Lx, Ly = self.Lx, self.Ly
        Vp = (Lx - 1) * (Ly - 1)

        for x in range(Lx - 1):
            for y in range(Ly - 1):
                xp = (x + 1)
                yp = (y + 1)
                w_U += np.real(U[x, y, 0] * U[xp, y, 1] * np.conj(U[x, yp, 0] * U[x, y, 1]))
                temp = np.vdot(z[x, y], z[xp, y]) * np.vdot(z[xp, y], z[xp, yp]) * np.vdot(z[xp, yp], z[x, yp]) * np.vdot(z[x, yp], z[x, y])
                w_z += np.cos(np.angle(temp))

        return w_U / Vp, w_z / Vp

    def get_boundary_z(self):
        # boundary z, starting at (0,0), and end at (0,0), return n_list with total 6*L-5 elements
        L = self.L
        z_list = np.concat([self.z[:-1, 0], self.z[2*L, :-1], np.flip(self.z[1:, L], axis=0), np.flip(self.z[0, 1:], axis=0)])
        return z_list
    
    def get_boundary_U(self):
        L = self.L
        U = self.U
        U_list = np.concat([U[:2*L, 0, 0], U[2*L, :L, 1], np.flip(np.conj(U[0:2*L, L, 0])), np.flip(np.conj(U[0, :L, 1]))])
        return U_list
    
    def get_boundary_coarse_conn(self):
        '''berry connection on boundary using U, return (6,) array'''
        L = self.L
        conn_list = np.zeros(6)
        conn_list[0] = np.angle(np.prod(self.U[:L, 0, 0]))
        conn_list[1] = np.angle(np.prod(self.U[L:2*L, 0, 0]))
        conn_list[2] = np.angle(np.prod(self.U[2*L, :L, 1]))
        conn_list[3] = -np.angle(np.prod(self.U[L:2*L, L, 0]))
        conn_list[4] = -np.angle(np.prod(self.U[:L, L, 0]))
        conn_list[5] = -np.angle(np.prod(self.U[0, :L, 1]))
        return conn_list
    
    def get_boundary_coarse_conn_z(self):
        '''berry connection on boundary using z, return (6,) array'''
        L = self.L
        conn_list = np.zeros(6)
        conn_list[0] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[:L, 0]), self.z[1:L+1, 0])))
        conn_list[1] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[L:-1, 0]), self.z[L+1:, 0])))
        conn_list[2] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[2*L, :-1]), self.z[2*L, 1:])))
        conn_list[3] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[L:-1, L]), self.z[L+1:, L])))
        conn_list[4] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[:L, L]), self.z[1:L+1, L])))
        conn_list[5] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[0, :-1]), self.z[0, 1:])))
        return U1_clip(conn_list)
        
    def get_conn(self):
        '''berry connection in the middle using U, [-pi, pi]'''
        L = self.L 
        return np.angle(np.prod(self.U[L, :-1, 1]))
    
    def get_conn_z(self):
        '''berry connection in the middle using z, [-pi, pi]'''
        L = self.L 
        z = self.z
        temp = np.sum(np.conj(z[L, :-1]) * z[L, 1:], axis=1)
        conn = np.sum(np.angle(temp))
        return U1_clip(conn)

class CPN_halfRefVil_HMCSampler_2plaq:
    """
    CP^(N-1) + U(1) sampler with global HMC updates, OBC with fixed boundary.

    System size: (2L+1) * (L+1) (2plaq)

    - z field: shape (Lx, Ly, N), complex, per-site |z|^2 = 1.
    - a field: shape (Lx, Ly, 2), real, U = exp(1j * a).
    - Boundary z and a are frozen (zero momentum, zero force) throughout
      the HMC trajectory.

    Refined half-Villainized action with beta1 * |z' z|^2 term.
    """

    def __init__(
        self,
        N,
        L,
        beta,
        beta1,
        alpha,
        z_list,
        U_list,
        epsilon=0.05,
        n_leapfrog=20,
        mass_a=1.0,
        mass_z=1.0,
        seed=None,
    ):
        self.L = L
        self.Lx = 2 * L + 1
        self.Ly = L + 1
        if N <= 1:
            raise ValueError("N too small")
        self.N = N
        self.beta = beta
        self.beta1 = beta1
        self.alpha = alpha

        if not (len(z_list) == 6 * L and len(U_list) == 6 * L):
            raise IndexError("length of z_list or U_list incorrect")
        self.z_list = z_list
        self.U_list = U_list

        self.epsilon = float(epsilon)
        self.n_leapfrog = int(n_leapfrog)
        self.mass_a = float(mass_a)
        self.mass_z = float(mass_z)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.n_leapfrog <= 0:
            raise ValueError("n_leapfrog must be positive")
        if self.mass_a <= 0 or self.mass_z <= 0:
            raise ValueError("mass_a and mass_z must be positive")

        if seed is not None:
            np.random.seed(seed)

        self.z = self._random_unit_vectors((self.Lx, self.Ly, self.N))
        self.a = 2 * np.pi * np.random.rand(self.Lx, self.Ly, 2)
        self._sync_U_from_a()

        self.boundary_mask_z = np.zeros((self.Lx, self.Ly), dtype=bool)
        self.boundary_mask_U = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self._init_config()
        self._wrap_phase_inplace()
        self._sync_U_from_a()
        self._normalize_z_inplace()

        self.accepted = 0
        self.attempted = 0

    @property
    def accept_rate(self):
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted

    # ------------------------------------------------------------
    # ---------- Initialization & Boundary -----------------------
    # ------------------------------------------------------------

    def _init_config(self):
        for x in range(self.Lx):
            for y in range(self.Ly):
                if x == 0 or x == self.Lx - 1 or y == 0 or y == self.Ly - 1:
                    self.boundary_mask_z[x, y] = True
                    self.z[x, y] = normalize_z(self._boundary_z(x, y, self.z_list))
        for x in range(self.Lx - 1):
            for y in (0, self.Ly - 1):
                self.boundary_mask_U[x, y, 0] = True
                self.a[x, y, 0] = np.angle(normalize_z(self._boundary_U(x, y, 0, self.U_list)))
        for y in range(self.Ly - 1):
            for x in (0, self.Lx - 1):
                self.boundary_mask_U[x, y, 1] = True
                self.a[x, y, 1] = np.angle(normalize_z(self._boundary_U(x, y, 1, self.U_list)))

    def _boundary_z(self, x, y, z_list):
        '''
        Generate the boundary z configuration according to z_list.
        '''
        L = self.L
        if y == 0:
            return z_list[x]
        elif x == 2 * L:
            return z_list[y + 2*L]
        elif y == L:
            return z_list[5*L - x]
        elif x == 0:
            return z_list[6*L - y]
        else:
            return self._random_unit_vectors(self.N)
        
    def _boundary_U(self, x, y, mu, U_list):
        '''
        Generate the boundary U configuration according to U_list.
        '''
        L = self.L
        if y == 0 and mu == 0:
            return U_list[x]
        elif x == 2 * L and mu == 1:
            return U_list[2*L + y]
        elif y == L and mu == 0:
            return np.conj(U_list[5*L - 1 - x])
        elif x == 0 and mu == 1:
            return np.conj(U_list[6*L - 1 - y])
        else:
            return np.exp(1j * 2 * np.pi * rand())

    # ------------------------------------------------------------
    # ---------- Internal utilities -------------------------------
    # ------------------------------------------------------------

    def _random_unit_vectors(self, shape):
        real = randn(*shape)
        imag = randn(*shape)
        v = real + 1j * imag
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm

    def _wrap_phase_inplace(self):
        self.a = (self.a + np.pi) % (2 * np.pi) - np.pi

    def _sync_U_from_a(self):
        self.U = np.exp(1j * self.a)

    def _normalize_z_inplace(self):
        norm = np.linalg.norm(self.z, axis=-1, keepdims=True)
        self.z /= norm

    def _project_tangent(self, z, field):
        overlap = np.einsum("ijk,ijk->ij", np.conj(z), field)
        return field - np.real(overlap)[:, :, None] * z

    # ------------------------------------------------------------
    # ---------- Action & Forces ----------------------------------
    # ------------------------------------------------------------

    def _spin_inner_x(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=0))

    def _spin_inner_y(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=1))

    def _plaquette(self):
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]
        plaq = np.zeros((self.Lx, self.Ly), dtype=complex)
        if self.Lx > 1 and self.Ly > 1:
            plaq[:-1, :-1] = (
                Ux[:-1, :-1]
                * Uy[1:, :-1]
                * np.conj(Ux[:-1, 1:] * Uy[:-1, :-1])
            )
        return plaq

    def action(self, mod=0):
        z = self.z
        U = self.U

        if self.Lx > 1:
            cp_term_x = np.real(U[:-1, :, 0] * vdot_z(z[1:, :, :], z[:-1, :, :]))
            if mod == 0:
                cp_term_1_x = (np.abs(vdot_z(z[1:, :, :], z[:-1, :, :])) ** 2 - 1) * self.N * self.beta1
            elif mod == 1:
                cp_term_1_x = np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(vdot_z(z[1:, :, :], z[:-1, :, :])) * 2 * self.N * self.beta1))
        if self.Ly > 1:
            cp_term_y = np.real(U[:, :-1, 1] * vdot_z(z[:, 1:, :], z[:, :-1, :]))
            if mod == 0:
                cp_term_1_y = (np.abs(vdot_z(z[:, 1:, :], z[:, :-1, :])) ** 2 - 1) * self.N * self.beta1
            elif mod == 1:
                cp_term_1_y = np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(vdot_z(z[:, 1:, :], z[:, :-1, :])) * 2 * self.N * self.beta1))

        plaq_term = np.real(self._plaquette())

        return -2.0 * self.N * self.beta * (np.sum(cp_term_x - 1) + np.sum(cp_term_y - 1)) - np.sum(cp_term_1_x) - np.sum(cp_term_1_y) - self.alpha * np.sum(plaq_term - 1)
    
    def _z_force(self, mod=0):
        """
        OBC force on z, then zeroed on boundary sites.
        """
        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        beta1 = self.beta1

        F = np.zeros_like(z)
        F1 = np.zeros_like(z)
        inner_x = self._spin_inner_x()
        inner_y = self._spin_inner_y()
        if self.Lx > 1:
            F[1:, :, :] += U[:-1, :, 0][:, :, None] * z[:-1, :, :]
            F[:-1, :, :] += np.conj(U[:-1, :, 0])[:, :, None] * z[1:, :, :]
            if mod == 0:
                F1[1:, :, :] += inner_x[:-1, :, None] * z[:-1, :, :] * beta1
                F1[:-1, :, :] += np.conj(inner_x)[:-1, :, None] * z[1:, :, :] * beta1
            elif mod == 1:
                inner_x_abs = np.abs(inner_x)
                temp = scipy.special.i1(inner_x_abs * 2 * N * beta1) / scipy.special.i0(inner_x_abs * 2 * N * beta1) * inner_x / inner_x_abs
                F1[1:, :, :] += temp[:-1, :, None] * z[:-1, :, :] * np.abs(beta1)
                F1[:-1, :, :] += np.conj(temp)[:-1, :, None] * z[1:, :, :] * np.abs(beta1)
        if self.Ly > 1:
            F[:, 1:, :] += U[:, :-1, 1][:, :, None] * z[:, :-1, :]
            F[:, :-1, :] += np.conj(U[:, :-1, 1])[:, :, None] * z[:, 1:, :]
            if mod == 0:
                F1[:, 1:, :] += inner_y[:, :-1, None] * z[:, :-1, :] * beta1
                F1[:, :-1, :] += np.conj(inner_y)[:, :-1, None] * z[:, 1:, :] * beta1
            elif mod == 1:
                inner_y_abs = np.abs(inner_y)
                temp = scipy.special.i1(inner_y_abs * 2 * N * beta1) / scipy.special.i0(inner_y_abs * 2 * N * beta1) * inner_y / inner_y_abs
                F1[:, 1:, :] += temp[:, :-1, None] * z[:, :-1, :] * np.abs(beta1)
                F1[:, :-1, :] += np.conj(temp)[:, :-1, None] * z[:, 1:, :] * np.abs(beta1)

        result = N * beta * F * 2.0 + N * F1 * 2.0
        result[self.boundary_mask_z] = 0.0
        return result

    def _link_force_complex(self):
        z = self.z

        F = np.zeros((self.Lx, self.Ly, 2), dtype=complex)

        F[:-1, :, 0] += 2.0 * self.N * self.beta * vdot_z(z[:-1, :, :], z[1:, :, :])
        F[:, :-1, 1] += 2.0 * self.N * self.beta * vdot_z(z[:, :-1, :], z[:, 1:, :])

        if self.alpha != 0.0 and self.Lx > 1 and self.Ly > 1:
            Ux = self.U[:, :, 0]
            Uy = self.U[:, :, 1]

            staple_x = np.zeros((self.Lx - 1, self.Ly), dtype=complex)
            staple_x[:, 1:] += (
                Uy[1:, :-1] * Ux[:-1, :-1] * np.conj(Uy[:-1, :-1])
            )
            staple_x[:, :-1] += (
                np.conj(Uy[1:, :-1]) * Ux[:-1, 1:] * Uy[:-1, :-1]
            )
            F[:-1, :, 0] += self.alpha * staple_x

            staple_y = np.zeros((self.Lx, self.Ly - 1), dtype=complex)
            staple_y[1:, :] += (
                Ux[:-1, 1:] * Uy[:-1, :-1] * np.conj(Ux[:-1, :-1])
            )
            staple_y[:-1, :] += (
                np.conj(Ux[:-1, 1:]) * Uy[1:, :-1] * Ux[:-1, :-1]
            )
            F[:, :-1, 1] += self.alpha * staple_y

        F[self.boundary_mask_U] = 0.0
        return F

    def _a_force(self):
        F_complex = self._link_force_complex()
        return np.imag(np.conj(self.U) * F_complex)

    # ------------------------------------------------------------
    # ---------- HMC trajectory -----------------------------------
    # ------------------------------------------------------------

    def _sample_momenta(self):
        p_a = np.sqrt(self.mass_a) * np.random.randn(self.Lx, self.Ly, 2)
        p_a[self.boundary_mask_U] = 0.0

        p_z = np.sqrt(self.mass_z) * (
            np.random.randn(self.Lx, self.Ly, self.N)
            + 1j * np.random.randn(self.Lx, self.Ly, self.N)
        )
        p_z[self.boundary_mask_z] = 0.0j
        p_z = self._project_tangent(self.z, p_z)
        return p_a, p_z

    def _kinetic(self, p_a, p_z):
        K_a = 0.5 * np.sum(p_a * p_a) / self.mass_a
        K_z = 0.5 * np.sum(np.abs(p_z) ** 2) / self.mass_z
        return K_a + K_z

    def hamiltonian(self, p_a, p_z, mod=0):
        return self.action(mod=mod) + self._kinetic(p_a, p_z)

    def hmc_step(self, epsilon=None, n_leapfrog=None, mod=0):
        eps = self.epsilon if epsilon is None else float(epsilon)
        nlf = self.n_leapfrog if n_leapfrog is None else int(n_leapfrog)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if nlf <= 0:
            raise ValueError("n_leapfrog must be positive")

        z_old = self.z.copy()
        a_old = self.a.copy()
        U_old = self.U.copy()

        p_a, p_z = self._sample_momenta()
        H_old = self.hamiltonian(p_a, p_z, mod=mod)

        force_a = self._a_force()
        force_z = self._project_tangent(self.z, self._z_force(mod=mod))
        p_a += 0.5 * eps * force_a
        p_z += 0.5 * eps * force_z
        p_z = self._project_tangent(self.z, p_z)

        for step in range(nlf):
            # update a and z
            self.a += eps * p_a / self.mass_a
            self._wrap_phase_inplace()
            self._sync_U_from_a()

            p_z_norm = np.linalg.norm(p_z, axis=-1)
            safe_mask = p_z_norm > 1e-15
            p_z_unit = np.zeros_like(p_z)
            p_z_unit[safe_mask] = (
                p_z[safe_mask] / p_z_norm[safe_mask, None]
            )
            cos_p = np.cos(eps * p_z_norm / self.mass_z)
            sin_p = np.sin(eps * p_z_norm / self.mass_z)
            p_z = cos_p[:,:,None] * p_z - sin_p[:,:,None] * self.z * p_z_norm[:, :, None]
            self.z = cos_p[:,:,None] * self.z + sin_p[:,:,None] * p_z_unit
            self._normalize_z_inplace()
            p_z = self._project_tangent(self.z, p_z)

            # update momenta
            force_a = self._a_force()
            force_z = self._project_tangent(self.z, self._z_force(mod=mod))
            coeff = 1.0 if step < nlf - 1 else 0.5
            p_a += coeff * eps * force_a
            p_z += coeff * eps * force_z
            p_z = self._project_tangent(self.z, p_z)

        H_new = self.hamiltonian(p_a, p_z, mod=mod)
        dH = H_new - H_old

        self.attempted += 1
        if np.log(np.random.rand()) < -dH:
            self.accepted += 1
            return True, dH

        self.z = z_old
        self.a = a_old
        self.U = U_old
        return False, dH

    def sweep(self, epsilon=None, n_leapfrog=None, mod=0):
        """Run one global HMC trajectory (alias kept for API familiarity)."""
        return self.hmc_step(epsilon=epsilon, n_leapfrog=n_leapfrog, mod=mod)

    # ------------------------------------------------------------
    # ---------- Measurements -------------------------------------
    # ------------------------------------------------------------

    def energy_density(self):
        """
        Compute (S / V / 2), where
          S = 2 * beta  * sum (1 - Re(z' U z))
            + beta1 * sum (1 - |z' z|^2 / 2)

        V = total number of links.
        Returns float.
        """
        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        beta1 = self.beta1
        Lx, Ly = self.Lx, self.Ly
        Vx = (Lx - 1) * Ly   # number of x-links
        Vy = Lx * (Ly - 1)   # number of y-links
        V = Vx + Vy

        E_beta = 0.0
        E_beta1 = 0.0

        for x in range(Lx - 1):
            for y in range(Ly):
                E_beta += 1 - np.real(U[x, y, 0] * np.vdot(z[x + 1, y], z[x, y]))
                E_beta1 += 0.5 - np.abs(np.vdot(z[x + 1, y], z[x, y]))**2 / 2

        for x in range(Lx):
            for y in range(Ly - 1):
                E_beta += 1 - np.real(U[x, y, 1] * np.vdot(z[x, y + 1], z[x, y]))
                E_beta1 += 0.5 - np.abs(np.vdot(z[x, y + 1], z[x, y]))**2 / 2

        return (beta * E_beta + beta1 * E_beta1) / (beta + beta1) / V

    def wilson_loop11(self):
        '''
        Compute average value of a 1*1 Wilson loop i.e. Re(UUUU) = cos(da).
        Use two methods: Re(UUUU) and cos(arg(ZZ) + arg(ZZ) + ...), and return two float.
        '''
        w_U = 0
        w_z = 0
        U = self.U
        z = self.z
        Lx, Ly = self.Lx, self.Ly
        Vp = (Lx - 1) * (Ly - 1)

        for x in range(Lx - 1):
            for y in range(Ly - 1):
                xp = (x + 1)
                yp = (y + 1)
                w_U += np.real(U[x, y, 0] * U[xp, y, 1] * np.conj(U[x, yp, 0] * U[x, y, 1]))
                temp = np.vdot(z[x, y], z[xp, y]) * np.vdot(z[xp, y], z[xp, yp]) * np.vdot(z[xp, yp], z[x, yp]) * np.vdot(z[x, yp], z[x, y])
                w_z += np.cos(np.angle(temp))

        return w_U / Vp, w_z / Vp

    def get_boundary_z(self):
        # boundary z, starting at (0,0), and end at (0,0), return n_list with total 6*L-5 elements
        L = self.L
        z_list = np.concat([self.z[:-1, 0], self.z[2*L, :-1], np.flip(self.z[1:, L], axis=0), np.flip(self.z[0, 1:], axis=0)])
        return z_list
    
    def get_boundary_U(self):
        L = self.L
        U = self.U
        U_list = np.concat([U[:2*L, 0, 0], U[2*L, :L, 1], np.flip(np.conj(U[0:2*L, L, 0])), np.flip(np.conj(U[0, :L, 1]))])
        return U_list
    
    def get_boundary_coarse_conn(self):
        '''berry connection on boundary using U, return (6,) array'''
        L = self.L
        conn_list = np.zeros(6)
        conn_list[0] = np.angle(np.prod(self.U[:L, 0, 0]))
        conn_list[1] = np.angle(np.prod(self.U[L:2*L, 0, 0]))
        conn_list[2] = np.angle(np.prod(self.U[2*L, :L, 1]))
        conn_list[3] = -np.angle(np.prod(self.U[L:2*L, L, 0]))
        conn_list[4] = -np.angle(np.prod(self.U[:L, L, 0]))
        conn_list[5] = -np.angle(np.prod(self.U[0, :L, 1]))
        return conn_list
    
    def get_boundary_coarse_conn_z(self):
        '''berry connection on boundary using z, return (6,) array'''
        L = self.L
        conn_list = np.zeros(6)
        conn_list[0] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[:L, 0]), self.z[1:L+1, 0])))
        conn_list[1] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[L:-1, 0]), self.z[L+1:, 0])))
        conn_list[2] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[2*L, :-1]), self.z[2*L, 1:])))
        conn_list[3] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[L:-1, L]), self.z[L+1:, L])))
        conn_list[4] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[:L, L]), self.z[1:L+1, L])))
        conn_list[5] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(self.z[0, :-1]), self.z[0, 1:])))
        return U1_clip(conn_list)
        
    def get_conn(self):
        '''berry connection in the middle using U, [-pi, pi]'''
        L = self.L 
        return np.angle(np.prod(self.U[L, :-1, 1]))
    
    def get_conn_z(self):
        '''berry connection in the middle using z, [-pi, pi]'''
        L = self.L 
        z = self.z
        temp = np.sum(np.conj(z[L, :-1]) * z[L, 1:], axis=1)
        conn = np.sum(np.angle(temp))
        return U1_clip(conn)

# ----------------------------------------Boundary z and U------------------------------------------
def rand_fine_z(L, N):
    '''Randomly generate z_list with shape (6*L, N)'''
    real = randn(6*L, N)
    imag = randn(6*L, N)
    v = real + 1j * imag
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v /= norm
    return v

def rand_fine_U(L):
    '''Randomly generate U_list with shape (6*L,)'''
    phases = 2 * np.pi * rand(6*L)
    return np.exp(1j * phases)

def from_z_to_U(z_list):
    '''Generate U_list using z_list'''
    U_list = np.einsum('ij,ij->i', np.conj(z_list), np.roll(z_list, shift=-1, axis=0))
    return U_list / np.abs(U_list)
    
def extract_param(L, z_list, U_list):
    '''
    Extract the info to write the Villain action from z_list and U_list (i.e. a1, a2, a3, a4, a5, a6, phi0, phi1, phi2, phi3, phi4, phi5), 
    connections are obtained by U or z

    :return a_list_U: (6,) array of connections obtained by U
    :return a_list_z: (6,) array of connections obtained by z
    :return phi_list: (6 * 2 * N,) array of phi obtained by z
    '''
    a_list_U = np.zeros(6)
    a_list_z = np.zeros(6)
    for i in range(6):
        a_list_U[i] = np.angle(np.prod(U_list[i*L:(i+1)*L]))
        a_list_z[i] = np.sum([np.angle(np.vdot(z_list[j + i*L], z_list[(j + i*L + 1) % (6*L)])) for j in range(L)])
    phi_list = np.hstack([from_z_to_phi(z_list[i*L]) for i in range(6)])
    return a_list_U, U1_clip(a_list_z), phi_list

# ----------------------------------------Data generation------------------------------------------
def run_halfRefVil_HMC(args):
    """
    run one HMC of half-refined Villain model and return the frequency of berry connection

    args = (N, L, beta, beta1, alpha, boundary_BC, boundary_pad, mod, boundary_n_therm, n_therm, n_meas, meas_interval, epsilon, n_leapfrog, mass_a, mass_z, bins)

    :return freq_U: freq of Berry connection obtained by U, shape (bins,)
    :return freq_z: freq of Berry connection obtained by z, shape (bins,)
    :return a_list_U: (6,) array of boundary connections obtained by U
    :return a_list_z: (6,) array of boundary connections obtained by z
    :return phi_list: (6*2*N,) array of boundary phi obtained by z
    :return acc_rate: HMC acceptance rate of the interior measurement sampler
                      (cumulative over its thermalization + measurement sweeps)
    """
    N, L, beta, beta1, alpha, boundary_BC, boundary_pad, mod, boundary_n_therm, n_therm, n_meas, meas_interval, epsilon, n_leapfrog, mass_a, mass_z, bins = args

    # boundary thermalization
    if boundary_BC == 'OBC':
        sampler0 = CPN_halfRefVil_HMCSampler_OBC(N=N, Lx=(2*L+1) + 2 * boundary_pad, Ly=L+1 + 2 * boundary_pad, beta=beta, beta1=beta1, alpha=alpha, epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z)
    elif boundary_BC == 'PBC':
        sampler0 = CPN_halfRefVil_HMCSampler(N=N, Lx=(2*L+1) + 2 * boundary_pad - 1, Ly=L+1 + 2 * boundary_pad - 1, beta=beta, beta1=beta1, alpha=alpha, epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z)
    else:
        raise Exception("boundary_BC should be 'OBC' or 'PBC'")
    if boundary_pad < 0:
        raise Exception("boundary_pad should be non-negative")
    for _ in range(boundary_n_therm):
        sampler0.sweep(mod=mod)
    z = sampler0.z
    U = sampler0.U
    if boundary_BC == 'OBC' or boundary_pad > 0:
        z_list = np.concat([z[boundary_pad:2*L+boundary_pad, boundary_pad], z[2*L+boundary_pad, boundary_pad:L+boundary_pad], np.flip(z[boundary_pad+1:2*L+1+boundary_pad, L+boundary_pad], axis=0), np.flip(z[boundary_pad, boundary_pad+1:L+1+boundary_pad], axis=0)])
        U_list = np.concat([U[boundary_pad:2*L+boundary_pad, boundary_pad, 0], U[2*L+boundary_pad, boundary_pad:L+boundary_pad, 1], np.flip(np.conj(U[boundary_pad:2*L+boundary_pad, L+boundary_pad, 0])), np.flip(np.conj(U[boundary_pad, boundary_pad:L+boundary_pad, 1]))])
    else:
        z_list = np.concat([z[:, 0], z[0, :], np.flip(np.roll(z[:, 0], -1, 0), axis=0), np.flip(np.roll(z[0, :], -1, 0), axis=0)])
        U_list = np.concat([U[:, 0, 0], U[0, :, 1], np.flip(np.conj(U[:, 0, 0])), np.flip(np.conj(U[0, :, 1]))])

    sampler = CPN_halfRefVil_HMCSampler_2plaq(N=N, L=L, beta=beta, beta1=beta1, alpha=alpha, z_list=z_list, U_list=U_list, epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z)
    a_list_U, a_list_z, phi_list = extract_param(L, z_list, U_list)

    conns_U = []
    conns_z = []

    # thermalize
    for _ in range(n_therm):
        sampler.sweep(mod=mod)

    # measurements
    for i in range(n_meas):
        sampler.sweep(mod=mod)
        if i % meas_interval == 0:
            conns_U.append(sampler.get_conn())
            conns_z.append(sampler.get_conn_z())

    conns_U = np.array(conns_U)
    freq_U, _ = np.histogram(conns_U, bins=bins, range=(-np.pi, np.pi), density=True)

    conns_z = np.array(conns_z)
    freq_z, _ = np.histogram(conns_z, bins=bins, range=(-np.pi, np.pi), density=True)

    return freq_U, freq_z, a_list_U, a_list_z, phi_list, sampler.accept_rate

if __name__ == "__main__":
    L = 2
    N = 2
    beta = 0.0
    beta1 = 1.0
    alpha = 0.0

    z_list = rand_fine_z(L, N)
    U_list = rand_fine_U(L)

    sampler = CPN_halfRefVil_HMCSampler_2plaq(N, L, beta, beta1, alpha, z_list=z_list, U_list=U_list, epsilon=0.02, n_leapfrog=50, seed=None)

    # inner_x = sampler._spin_inner_x()
    # inner_x_abs = np.abs(inner_x)
    # temp = scipy.special.i1(inner_x_abs * 2 * N * beta1) / scipy.special.i0(inner_x_abs * 2 * N * beta1) * inner_x / inner_x_abs
    # print(temp)
    acc, dH = [], []
    for _ in range(10):
        acc_t, dH_t = sampler.sweep(mod=1)
        acc.append(acc_t)
        dH.append(dH_t)
    
    print(acc)
    print(dH)