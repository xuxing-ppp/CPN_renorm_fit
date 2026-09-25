import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from numpy.random import rand, randn
from func.func_CPN_Original import CPNSampler, CPNSampler_OBC

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

def vdot_z(z1, z2):
    return np.einsum('ijk,ijk->ij', np.conj(z1), z2)

def U1_clip(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


class CPNSampler_1plaq(CPNSampler_OBC):
    """
    Clean CP^(N-1) Over-Heatbath Monte-Carlo sampler for a single renormalized
    plaquette, OBC with a fixed (frozen) boundary. No integer Villain field s.

    System size: (L+1) * (L+1)  (1 renormalized plaquette of renorm scale L).

    - z field: shape (Lx, Ly, N), complex, per-site |z|^2 = 1.
    - U field: shape (Lx, Ly, 2), complex unit phase (U = exp(1j*a)).
    - Boundary z and U are frozen throughout the sweep (skipped via the
      boundary masks); only the interior is heatbath/overrelaxation-updated.

    Action ("half-Villainized", cos(da) plaquette, no integer variable):
      -2N*beta * sum (Re(z' U z) - 1)
      - alpha1 * sum (cos(da) - 1)
    """

    def __init__(self, N, L, beta, alpha1, z_list, U_list, seed=None):
        self.L = L
        self.Lx = L + 1
        self.Ly = L + 1
        if N <= 1:
            raise ValueError("N too small")
        self.N = N
        self.beta = beta
        self.alpha1 = alpha1

        if not (len(z_list) == 4 * L and len(U_list) == 4 * L):
            raise IndexError("length of z_list or U_list incorrect")
        self.z_list = z_list
        self.U_list = U_list

        if seed is not None:
            np.random.seed(seed)

        self.z = self._random_unit_vectors((self.Lx, self.Ly, self.N))
        phases = 2 * np.pi * rand(self.Lx, self.Ly, 2)
        self.U = np.exp(1j * phases)

        self.boundary_mask_z = np.zeros((self.Lx, self.Ly), dtype=bool)
        self.boundary_mask_U = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self._init_config()

    # ------------------------------------------------------------
    # ---------- Initialization & Boundary -----------------------
    # ------------------------------------------------------------
    # Boundary is a single CCW loop of 4*L sites and 4*L links, indexed
    #   bottom (y=0, +x) : 0 .. L
    #   right  (x=L, +y) : L+1 .. 2L
    #   top    (y=L, -x) : 2L+1 .. 3L
    #   left   (x=0, -y) : 3L+1 .. 4L-1
    # ------------------------------------------------------------

    def _init_config(self):
        L = self.L
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

    def _boundary_z(self, x, y, z_list):
        '''
        Generate the boundary z configuration according to z_list.
        '''
        L = self.L
        if y == 0:
            return z_list[x]                 # bottom, x=0..L
        elif x == L:
            return z_list[L + y]             # right, y=1..L  -> L+1..2L
        elif y == L:
            return z_list[3 * L - x]         # top, x=L-1..0  -> 2L+1..3L
        elif x == 0:
            return z_list[4 * L - y]         # left, y=L-1..1 -> 3L+1..4L-1
        else:
            return self._random_unit_vectors(self.N)

    def _boundary_U(self, x, y, mu, U_list):
        '''
        Generate the boundary U configuration according to U_list.
        '''
        L = self.L
        if y == 0 and mu == 0:
            return U_list[x]                       # bottom, +x, x=0..L-1 -> 0..L-1
        elif x == L and mu == 1:
            return U_list[L + y]                   # right,  +y, y=0..L-1 -> L..2L-1
        elif y == L and mu == 0:
            return np.conj(U_list[3 * L - 1 - x])  # top, -x, x=0..L-1 -> 2L..3L-1
        elif x == 0 and mu == 1:
            return np.conj(U_list[4 * L - 1 - y])  # left, -y, y=0..L-1 -> 3L..4L-1
        else:
            return np.exp(1j * 2 * np.pi * rand())

    # ------------------------------------------------------------
    # ---------- MC Sweep (skip the frozen boundary) -------------
    # ------------------------------------------------------------

    def sweep(self, heatbath_fraction=0.5):
        """Heatbath + overrelaxation sweep; boundary sites/links are skipped."""
        # mu = -1 stands for z update. The x/y ranges stop at Lx-2/Ly-2 so we
        # never attempt the non-existent mu=0 link at x=Lx-1 or mu=1 link at
        # y=Ly-1; those, together with the y=0 / x=0 boundary links and the
        # x=0,y=0 boundary sites, are frozen and skipped via the masks.
        indices = [(x, y, mu) for y in range(self.Ly - 1)
                           for x in range(self.Lx - 1)
                           for mu in (-1, 0, 1)]
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

    # def get_vortex(self):
    #     '''
    #     Total integer winding through the single plaquette.
    #     Use 3 methods: U, z and s. The Original model carries no integer
    #     Villain field s, so vortex_s is identically zero.

    #     :return vortex: (3,) array of integer vortex from U, z and s
    #     '''
    #     L = self.L
    #     a = np.angle(self.U)          # recover the continuous phase from U
    #     z = self.z

    #     temp_U = (a[:, :, 0] + np.roll(a, -1, axis=0)[:, :, 1]
    #         - np.roll(a, -1, axis=1)[:, :, 0] - a[:, :, 1]) / 2 / np.pi
    #     vU = -np.sum(np.round(temp_U[:L, :L]))
    #     temp_z1 = (
    #         np.angle(vdot_z(z, np.roll(z, -1, 0)))
    #         + np.angle(vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1))))
    #         + np.angle(vdot_z(np.roll(z, -1, (0, 1)), z))
    #     ) / 2 / np.pi
    #     temp1 = (
    #         np.angle(vdot_z(z, np.roll(z, -1, (0, 1))))
    #         + np.angle(vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1)))
    #         + np.angle(vdot_z(np.roll(z, -1, 1), z))
    #     ) / 2 / np.pi
    #     vz = -np.sum(np.round(temp_z1[:L, :L]) + np.round(temp1[:L, :L]))
    #     vs = 0                         # Original model has no integer s field
    #     return np.array([vU, vz, vs])
    
    def get_topo(self):
        '''
        Total topo charge through the single plaquette.
        Use 2 methods: U, z

        :return topo: (2,) array of integer topo charge from U, z
        '''
        L = self.L
        a = np.angle(self.U)
        z = self.z

        temp_U = (a[:, :, 0] + np.roll(a, -1, axis=0)[:, :, 1]
            - np.roll(a, -1, axis=1)[:, :, 0] - a[:, :, 1]) / 2 / np.pi
        qU = np.sum(temp_U[:L, :L] - np.round(temp_U[:L, :L]))
        temp_z1 = np.angle(
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), z)
        ) / 2 / np.pi
        temp_z2 = np.angle(
            vdot_z(z, np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        ) / 2 / np.pi
        qz = np.sum(temp_z1[:L, :L] + temp_z2[:L, :L])
        return np.array([qU, qz])

    def get_boundary_z(self):
        '''boundary z, starting at (0,0), CCW; total 4*L elements.'''
        L = self.L
        z = self.z
        return np.concat([
            z[:L, 0],                # bottom, x=0..L-1
            z[L, :L],                # right,  y=0..L-1
            np.flip(z[1:L + 1, L], axis=0),   # top, x=L..1
            np.flip(z[0, 1:L + 1], axis=0),   # left, y=L..1
        ])

    def get_boundary_U(self):
        '''boundary U, starting at (0,0), CCW; total 4*L elements.'''
        L = self.L
        U = self.U
        return np.concat([
            U[:L, 0, 0],                          # bottom, +x
            U[L, :L, 1],                          # right,  +y
            np.flip(np.conj(U[:L, L, 0]), axis=0),  # top, -x (conj + flip)
            np.flip(np.conj(U[0, :L, 1]), axis=0),  # left, -y (conj + flip)
        ])

    def get_boundary_coarse_conn(self):
        '''Berry connection on the 4 coarse boundary links using U; shape (4,).'''
        L = self.L
        conn = np.zeros(4)
        conn[0] = np.angle(np.prod(self.U[:L, 0, 0]))       # bottom, +x
        conn[1] = np.angle(np.prod(self.U[L, :L, 1]))       # right,  +y
        conn[2] = -np.angle(np.prod(self.U[:L, L, 0]))      # top,    -x
        conn[3] = -np.angle(np.prod(self.U[0, :L, 1]))      # left,   -y
        return conn

    def get_boundary_coarse_conn_z(self):
        '''Berry connection on the 4 coarse boundary links using z; shape (4,).'''
        L = self.L
        z = self.z
        conn = np.zeros(4)
        conn[0] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(z[:L, 0]), z[1:L + 1, 0])))   # bottom, +x
        conn[1] = np.sum(np.angle(np.einsum('ij,ij->i', np.conj(z[L, :L]), z[L, 1:L + 1])))    # right,  +y
        conn[2] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(z[:L, L]), z[1:L + 1, L])))   # top,    -x
        conn[3] = -np.sum(np.angle(np.einsum('ij,ij->i', np.conj(z[0, :L]), z[0, 1:L + 1])))   # left,   -y
        return U1_clip(conn)

    def get_boundary_coarse_phi(self):
        '''phi at the 4 coarse corners (CCW from (0,0)); shape (4*2*N,).'''
        L = self.L
        phi = np.zeros((4, 2 * self.N))
        phi[0] = from_z_to_phi(self.z[0, 0])
        phi[1] = from_z_to_phi(self.z[L, 0])
        phi[2] = from_z_to_phi(self.z[L, L])
        phi[3] = from_z_to_phi(self.z[0, L])
        return phi.reshape(-1)


# ----------------------------------------Boundary z and U------------------------------------------
def rand_fine_z(L, N):
    '''Randomly generate z_list with shape (4*L, N).'''
    real = randn(4 * L, N)
    imag = randn(4 * L, N)
    v = real + 1j * imag
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v /= norm
    return v

def rand_fine_U(L):
    '''Randomly generate U_list with shape (4*L,).'''
    phases = 2 * np.pi * rand(4 * L)
    return np.exp(1j * phases)

def from_z_to_U(z_list):
    '''Generate U_list using z_list (CCW nearest-neighbour products).'''
    U_list = np.einsum('ij,ij->i', np.conj(z_list), np.roll(z_list, -1, axis=0))
    return U_list / np.abs(U_list)

def extract_param(L, z_list, U_list):
    '''
    Extract the coarse-plaquette parameters from z_list and U_list
    (i.e. a0..a3 on the 4 boundary links, phi0..phi3 at the 4 corners).

    :return a_list_U: (4,) array of connections obtained by U
    :return a_list_z: (4,) array of connections obtained by z
    :return phi_list: (4 * 2 * N,) array of phi obtained by z
    '''
    a_list_U = np.zeros(4)
    a_list_z = np.zeros(4)
    for i in range(4):
        a_list_U[i] = np.angle(np.prod(U_list[i * L:(i + 1) * L]))
        a_list_z[i] = np.sum([np.angle(np.vdot(z_list[j + i * L], z_list[(j + i * L + 1) % (4 * L)])) for j in range(L)])
    phi_list = np.hstack([from_z_to_phi(z_list[i * L]) for i in range(4)])
    return a_list_U, U1_clip(a_list_z), phi_list


# ----------------------------------------Data generation------------------------------------------
def run_Original(args):
    """
    Run one heatbath MC of the Original (no-s) 1-plaquette model and return the
    integer-vortex / Berry-flux measurements together with the coarse boundary
    parameters. Same return contract as run_RefVil_HMC, with vortex_s == 0.

    args = (N, L, beta, alpha1, boundary_BC, boundary_pad,
            boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas)

    :return v_array: measured vortex numbers, shape (n_meas_kept, 3) = [vU, vz, 0]
    :return da_U:    flux through the coarse plaquette from U, scalar
    :return da_z:    flux through the coarse plaquette from z, scalar
    :return acc_hmc:   None (heatbath sampler -- no HMC acceptance rate)
    :return acc_metro: None (no s-Metropolis in the Original action)
    """
    (N, L, beta, alpha1, boundary_BC, boundary_pad,
     boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas) = args

    # ---------- boundary thermalization ----------
    if boundary_BC == 'OBC':
        sampler0 = CPNSampler_OBC(
            N=N, Lx=(L + 1) + 2 * boundary_pad, Ly=(L + 1) + 2 * boundary_pad,
            beta=beta, alpha1=alpha1)
    elif boundary_BC == 'PBC':
        sampler0 = CPNSampler(
            N=N, Lx=(L + 1) + 2 * boundary_pad - 1, Ly=(L + 1) + 2 * boundary_pad - 1,
            beta=beta, alpha1=alpha1)
    else:
        raise Exception("boundary_BC should be 'OBC' or 'PBC'")
    if boundary_pad < 0:
        raise Exception("boundary_pad should be non-negative")

    for _ in range(boundary_n_therm):
        sampler0.sweep()

    z = sampler0.z
    U = sampler0.U
    p = boundary_pad
    # extract the 4*L CCW boundary loop from the thermalized patch
    if boundary_BC == 'OBC' or boundary_pad > 0:
        z_list = np.concat([
            z[p:L + p, p],                         # bottom, x=0..L-1
            z[L + p, p:L + p],                     # right,  y=0..L-1
            np.flip(z[p + 1:L + 1 + p, L + p], axis=0),   # top, x=L..1
            np.flip(z[p, p + 1:L + 1 + p], axis=0),       # left, y=L..1
        ])
        U_list = np.concat([
            U[p:L + p, p, 0],                              # bottom, +x
            U[L + p, p:L + p, 1],                          # right,  +y
            np.flip(np.conj(U[p:L + p, L + p, 0])),        # top, -x
            np.flip(np.conj(U[p, p:L + p, 1])),            # left, -y
        ])
    else:
        # PBC with zero padding: identify the boundary with the first row/col
        z_list = np.concat([
            z[:L, 0], z[L, :L],
            np.flip(z[1:L + 1, L], axis=0),
            np.flip(z[0, 1:L + 1], axis=0),
        ])
        U_list = np.concat([
            U[:L, 0, 0], U[L, :L, 1],
            np.flip(np.conj(U[:L, L, 0])),
            np.flip(np.conj(U[0, :L, 1])),
        ])

    sampler = CPNSampler_1plaq(
        N=N, L=L, beta=beta, alpha1=alpha1, z_list=z_list, U_list=U_list)
    a_list_U, a_list_z, _ = extract_param(L, z_list, U_list)
    da_U = np.sum(a_list_U)
    da_z = np.sum(a_list_z)

    # ---------- thermalize interior ----------
    for _ in range(n_therm):
        sampler.sweep(heatbath_fraction=hf_therm)

    # ---------- measurements ----------
    # v_array = []
    q_array = []
    for i in range(n_meas):
        sampler.sweep(heatbath_fraction=hf_meas)
        if i % meas_interval == 0:
            # v_array.append(sampler.get_vortex())
            q_array.append(sampler.get_topo())
    q_array = np.array(q_array)  # shape (n_meas, 2)
    v_array_temp = q_array - np.array([da_U, da_z])[None, :] / 2 / np.pi

    # Original: vs = 0
    v_array = np.concatenate([v_array_temp, np.zeros((v_array_temp.shape[0], 1))], axis=1)

    return v_array, da_U, da_z, None, None


def get_freq_v(v_array, zero_pad=2):
    v_array = np.array(v_array)
    bins = np.arange(np.min(v_array) - 0.5 - int(zero_pad), np.max(v_array) + 0.6 + int(zero_pad), 1.0)
    freq_v, _ = np.histogram(v_array, bins=bins, density=True)
    v_list = np.round((bins[:-1] + bins[1:]) / 2).astype(int)  # bin centers
    return v_list, freq_v

def get_X_freq(v_array, da, zero_pad=2):
    v_array = np.array(v_array)
    v_list, freq_v = get_freq_v(v_array, zero_pad=zero_pad)
    X = np.stack([v_list, da * np.ones_like(v_list)], axis=-1)
    return X, freq_v

if __name__ == "__main__":
    # L = 4
    # N = 2
    # beta = 1.0
    # alpha1 = 1.0

    # z_list = rand_fine_z(L, N)
    # U_list = rand_fine_U(L)

    # sampler = CPNSampler_1plaq(
    #     N=N, L=L, beta=beta, alpha1=alpha1,
    #     z_list=z_list, U_list=U_list, seed=None)

    # # boundary round-trip check: extract_param must match the sampler's own coarse view
    # a_list_U, a_list_z, _ = extract_param(L, z_list, U_list)
    # print("z  boundary round-trip :", np.linalg.norm(z_list - sampler.get_boundary_z()))
    # print("U  boundary round-trip :", np.linalg.norm(U_list - sampler.get_boundary_U()))
    # print("aU round-trip          :", np.linalg.norm(a_list_U - sampler.get_boundary_coarse_conn()))
    # print("aZ round-trip          :", np.linalg.norm(a_list_z - sampler.get_boundary_coarse_conn_z()))

    # # a few sweeps + a vortex sample
    # for _ in range(10):
    #     sampler.sweep()
    # print("vortex sample [vU, vz, vs]:", sampler.get_vortex())
    args = (2, 4, 0.5, 0.3, 'OBC', 2, 10, 10, 10, 1, 0.5, 0.5)
    v_array, da_U, da_z, acc_hmc, acc_metro = run_Original(args)
    print("v_array:", v_array)
    print("acc_hmc:", acc_hmc, "acc_metro:", acc_metro)