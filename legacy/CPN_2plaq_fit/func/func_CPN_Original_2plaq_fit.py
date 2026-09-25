import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from numpy.random import rand, randn, vonmises
from func.func_CPN_Original import CPNSampler_OBC, CPNSampler

def random_unit_vectors(shape):
    """
    Samples random complex unit vectors uniformly on S^{2N-1}.
    shape = (L1,L2,...,N)
    """
    real = randn(*shape)
    imag = randn(*shape)
    v = real + 1j * imag
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm

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

def U1_clip(x):
    return np.mod(x + np.pi, 2*np.pi) - np.pi

class CPNSampler_2plaq:
    """
    Clean CP^(N-1) Over-Heatbath Monte-Carlo sampler, OBC.

    System size: (2L+1) * (L+1) (2plaq)

    N-dim normalized complex vector on sites, and U(1) phases on links.

    Action is "half-Villainized" i.e. with cos(da) term but no integer variable.

    Boundary is fixed by given z, while a is fixed by saddle: arg(z vdot z')
    """

    def __init__(self, N, L, beta, alpha, z_list, U_list, seed=None):
        self.L = L
        self.Lx = 2*L + 1
        self.Ly = L + 1
        if N <= 1:
            raise ValueError("N too small")
        self.N = N
        self.beta = beta
        self.alpha = alpha
        if not(len(z_list) == 6*L and len(U_list) == 6*L):
            raise IndexError("length of z_list or U_list incorrect")
        self.z_list = z_list
        self.U_list = U_list

        if seed is not None:
            np.random.seed(seed)

        # Randomly initialize z and U, boundary is fixed by z_list and U_list
        # z-field: shape (Lx,Ly,N) complex
        self.z = self._random_unit_vectors((self.Lx, self.Ly, self.N))
        # U(1) link field: shape (Lx,Ly,2) complex of unit norm
        # dir=0 → x direction, dir=1 → y direction
        phases = 2*np.pi * rand(self.Lx, self.Ly, 2)
        self.U = np.exp(1j * phases)

        self.boundary_mask_z = np.zeros((self.Lx, self.Ly), dtype=bool)
        self.boundary_mask_U = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self._init_config()

    def _init_config(self):
        for x in range(self.Lx):
            for y in range(self.Ly):
                if x == 0 or x == self.Lx - 1 or y == 0 or y == self.Ly - 1:
                    self.boundary_mask_z[x, y] = True
                    self.z[x, y] = normalize_z(self._boundary_z(x, y, self.z_list))
        for x in range(self.Lx - 1):
            for y in (0, self.Ly - 1):
                self.boundary_mask_U[x, y, 0] = True
                self.U[x, y, 0] = normalize_z(self._boundary_U(x, y, 0, self.U_list))
        for y in range(self.Ly - 1):
            for x in (0, self.Lx - 1):
                self.boundary_mask_U[x, y, 1] = True
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
        Compute local force F_z(x, y) = 2N*beta sum_mu lambda*(x,mu) z(x+mu) + lambda(x-mu,mu) z(x-mu)
        Returns shape (N,) complex.
        """
        if x == 0 or x == self.Lx - 1 or y == 0 or y == self.Ly - 1:
            raise Exception("boundary update")

        # neighbors with OBC
        xp = (x + 1)
        xm = (x - 1)
        yp = (y + 1)
        ym = (y - 1)

        z = self.z
        U = self.U

        Fx = U[xm, y, 0] * z[xm, y] + np.conj(U[x, y, 0]) * z[xp, y]
        Fy = U[x, ym, 1] * z[x, ym] + np.conj(U[x, y, 1]) * z[x, yp]

        return 2 * self.N * self.beta * (Fx + Fy)

    def force_U(self, x, y, mu):
        """
        Force for U(1) link:
        F = 2N*beta z(x+mu) dot conj(z(x)) + alpha * (plaq terms)
        return a complex number
        """
        if (x == 0 and mu == 1) or x == self.Lx - 1 or (y == 0 and mu == 0) or y == self.Ly - 1:
            raise Exception("boundary update")
        
        # neighbors with OBC
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
            # no meaningful force → random unit vector
            self.z[x, y] = self._random_unit_vectors((1, self.N))[0]
            return

        rnorm = 0
        while rnorm < 1e-10:
            # generate perpendicular random direction
            r = randn(self.N) + 1j*randn(self.N)
            # project out componeLy along F
            r -= (np.real(np.vdot(F, r)) / Fnorm**2) * F
            rnorm = np.linalg.norm(r)
        
        r /= rnorm

        # Sample angle θ from correct distribution:
        # P(θ) ∝ sin^(2N-2)(θ) * exp( |F| cos θ )
        # For z, N > 1
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
        Over-heatbath update for U(1) link.
        Sample angle from distribution:
            P(theta) ∝ exp( |F| cos(theta - arg(F)) )
        where F = < z(x+mu), z(x) >
        """
        F = self.force_U(x, y, mu)
        Fmag = np.abs(F)

        if Fmag < 1e-10:
            self.U[x, y, mu] = np.exp(1j * 2*np.pi * rand())
            return

        # Sample Δθ relative to arg(F):
        # P(Δ) ∝ exp( |F| cos(theta - arg(F)) )
        # This is the Von Mises distribution.
        new_phase = vonmises(np.angle(F), Fmag)
        self.U[x, y, mu] = np.exp(1j * new_phase)

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

    # ------------------------------------------------------------
    # ---------- MC Sweep -----------------------------------------
    # ------------------------------------------------------------

    def sweep(self, heatbath_fraction=0.5):
        """Heatbath + overrelaxation sweep."""
        # mu = -1 stands for z update
        # OBC
        indices = [(x, y, mu) for y in range(self.Ly - 1) for x in range(self.Lx - 1) for mu in (-1, 0, 1)]
        np.random.shuffle(indices)
        for (x, y, mu) in indices:
            if mu == -1:
                if self.boundary_mask_z[x, y]:
                    continue
                if rand() < heatbath_fraction:
                    self.hb_update_z(x, y)
                else:
                    self.or_update_z(x, y)
            else:
                if self.boundary_mask_U[x, y, mu]:
                    continue
                if rand() < heatbath_fraction:
                    self.hb_update_U(x, y, mu)
                else:
                    self.or_update_U(x, y, mu)

    # ------------------------------------------------------------
    # ---------- Measurements -------------------------------------
    # ------------------------------------------------------------

    def energy_density(self):
        """
        Compute S/N(link)/2, S = 2Nβ sum (1 - Re(z' U z)).
        Returns float.
        """
        E = 0.0
        z = self.z
        U = self.U
        Lx, Ly = self.Lx, self.Ly
        N_links = (Lx - 1) * Ly + (Ly - 1) * Lx

        for x in range(Lx - 1):
            for y in range(Ly - 1):
                xp = (x + 1)
                yp = (y + 1)
                E += (1 - np.real(U[x, y, 0] * np.vdot(z[xp, y], z[x, y])))
                E += (1 - np.real(U[x, y, 1] * np.vdot(z[x, yp], z[x, y])))
        for x in range(Lx - 1):
            E += (1 - np.real(U[x, Ly-1, 0] * np.vdot(z[x+1, Ly-1], z[x, Ly-1])))
        for y in range(Ly - 1):
            E += (1 - np.real(U[Lx-1, y, 1] * np.vdot(z[Lx-1, y+1], z[Lx-1, y])))

        return E / N_links
    
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
        V = (Lx - 1) * (Ly - 1)

        for x in range(Lx - 1):
            for y in range(Ly - 1):
                xp = (x + 1)
                yp = (y + 1)
                w_U += np.real(U[x, y, 0] * U[xp, y, 1] * np.conj(U[x, yp, 0] * U[x, y, 1]))
                temp = np.vdot(z[x, y], z[xp, y]) * np.vdot(z[xp, y], z[xp, yp]) * np.vdot(z[xp, yp], z[x, yp]) * np.vdot(z[x, yp], z[x, y])
                w_z += np.cos(np.angle(temp))
        
        return w_U / V, w_z / V
    
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

# ----------------------------------------Fitting and Testing-----------------------------------
def run_Original(args):
    """
    run one HB MC of original model and return the frequency of berry connection

    args = (N, L, beta, alpha, boundary_BC, boundary_pad, boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas, bins)

    :return freq_U: freq of Berry connection obtained by U, shape (bins,)
    :return freq_z: freq of Berry connection obtained by z, shape (bins,)
    :return a_list_U: (6,) array of boundary connections obtained by U
    :return a_list_z: (6,) array of boundary connections obtained by z
    :return phi_list: (6*2*N,) array of boundary phi obtained by z
    :return acc_rate: None (heatbath sampler -- no HMC acceptance rate)
    """
    N, L, beta, alpha, boundary_BC, boundary_pad, boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas, bins = args

    # boundary thermalization
    if boundary_BC == 'OBC':
        sampler0 = CPNSampler_OBC(N=N, Lx=(2*L+1) + 2 * boundary_pad, Ly=L+1 + 2 * boundary_pad, beta=beta, alpha=alpha)
    elif boundary_BC == 'PBC':
        sampler0 = CPNSampler(N=N, Lx=(2*L+1) + 2 * boundary_pad - 1, Ly=L+1 + 2 * boundary_pad - 1, beta=beta, alpha=alpha)
    else:
        raise Exception("boundary_BC should be 'OBC' or 'PBC'")
    if boundary_pad < 0:
        raise Exception("boundary_pad should be non-negative")
    for _ in range(boundary_n_therm):
        sampler0.sweep()
    z = sampler0.z
    U = sampler0.U
    if boundary_BC == 'OBC' or boundary_pad > 0:
        z_list = np.concat([z[boundary_pad:2*L+boundary_pad, boundary_pad], z[2*L+boundary_pad, boundary_pad:L+boundary_pad], np.flip(z[boundary_pad+1:2*L+1+boundary_pad, L+boundary_pad], axis=0), np.flip(z[boundary_pad, boundary_pad+1:L+1+boundary_pad], axis=0)])
        U_list = np.concat([U[boundary_pad:2*L+boundary_pad, boundary_pad, 0], U[2*L+boundary_pad, boundary_pad:L+boundary_pad, 1], np.flip(np.conj(U[boundary_pad:2*L+boundary_pad, L+boundary_pad, 0])), np.flip(np.conj(U[boundary_pad, boundary_pad:L+boundary_pad, 1]))])
    else:
        z_list = np.concat([z[:, 0], z[0, :], np.flip(np.roll(z[:, 0], -1, 0), axis=0), np.flip(np.roll(z[0, :], -1, 0), axis=0)])
        U_list = np.concat([U[:, 0, 0], U[0, :, 1], np.flip(np.conj(U[:, 0, 0])), np.flip(np.conj(U[0, :, 1]))])

    sampler = CPNSampler_2plaq(N=N, L=L, beta=beta, alpha=alpha, z_list=z_list, U_list=U_list)
    a_list_U, a_list_z, phi_list = extract_param(L, z_list, U_list)

    conns_U = []
    conns_z = []

    # thermalize
    for _ in range(n_therm):
        sampler.sweep(heatbath_fraction=hf_therm)

    # measurements
    for i in range(n_meas):
        sampler.sweep(heatbath_fraction=hf_meas)
        if i % meas_interval == 0:
            conns_U.append(sampler.get_conn())
            conns_z.append(sampler.get_conn_z())

    conns_U = np.array(conns_U)
    freq_U, _ = np.histogram(conns_U, bins=bins, range=(-np.pi, np.pi), density=True)

    conns_z = np.array(conns_z)
    freq_z, _ = np.histogram(conns_z, bins=bins, range=(-np.pi, np.pi), density=True)

    return freq_U, freq_z, a_list_U, a_list_z, phi_list, None

if __name__ == '__main__':
    N = 2
    L = 2
    beta = 0.5
    alpha = 0.5
    z_list = rand_fine_z(L, N)
    U_list = rand_fine_U(L)
    sampler = CPNSampler_2plaq(L=L, N=N, beta=beta, alpha=alpha, z_list=z_list, U_list=U_list)
    print(np.linalg.norm(z_list - sampler.get_boundary_z()))
    print(np.linalg.norm(U_list - sampler.get_boundary_U()))
    a_list_U, a_list_z, phi_list = extract_param(L, z_list, U_list)
    print(np.linalg.norm(a_list_U - sampler.get_boundary_coarse_conn()))
    print(a_list_z)
    print(sampler.get_boundary_coarse_conn_z())
    print(np.linalg.norm(a_list_z - sampler.get_boundary_coarse_conn_z()))