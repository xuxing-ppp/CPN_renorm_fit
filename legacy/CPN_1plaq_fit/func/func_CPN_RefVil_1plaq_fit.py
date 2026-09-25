import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import scipy
from numpy.random import rand, randn
from func.func_CPN_RefVil_HMC import CPN_RefVil_HMCSampler, CPN_RefVil_HMCSampler_OBC

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
    phi0 = phi.reshape(-1, 2)
    return phi0[:, 0] + 1j * phi0[:, 1]

def vdot_z(z1, z2):
    return np.einsum('ijk,ijk->ij', np.conj(z1), z2)

def U1_clip(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


class CPN_RefVil_HMCSampler_1plaq(CPN_RefVil_HMCSampler_OBC):
    """
    CP^(N-1) + U(1) + Villain-integer s sampler with global HMC updates,
    OBC with a fixed boundary.

    System size: (L+1) * (L+1)  (1 renormalized plaquette of renorm scale L).

    - z field: shape (Lx, Ly, N), complex, per-site |z|^2 = 1.
    - a field: shape (Lx, Ly, 2), real, U = exp(1j * a).
    - s field: shape (Lx, Ly), integer; lives on the L*L physical plaquettes
      s[:L, :L] (i.e. s[:-1, :-1]).
    - Boundary z and a are frozen (zero momentum, zero force) throughout the
      HMC trajectory.

    Action (RefVil, half-Villainized matter + full Villain plaquette):
      -2N*beta * sum (Re(z' U z) - 1)
      - sign(beta1)*sum log I0(2N*beta1*|z' z|)            [mod = 1]
      or  -N*beta1 * sum (|z' z|^2 - 1)                   [mod = 0]
      + alpha/2   * sum (da + 2*pi*s)^2
      - alpha1    * sum (cos(da + 2*pi*s) - 1)
    """

    def __init__(
        self,
        N,
        L,
        beta,
        beta1,
        alpha,
        alpha1,
        z_list,
        U_list,
        epsilon=0.05,
        n_leapfrog=20,
        mass_a=1.0,
        mass_z=1.0,
        s_step=0.5,
        s_update_num=1,
        seed=None,
    ):
        self.L = L
        self.Lx = L + 1
        self.Ly = L + 1
        if N <= 1:
            raise ValueError("N too small")
        self.N = N
        self.beta = beta
        self.beta1 = beta1
        self.alpha = alpha
        self.alpha1 = alpha1

        if not (len(z_list) == 4 * L and len(U_list) == 4 * L):
            raise IndexError("length of z_list or U_list incorrect")
        self.z_list = z_list
        self.U_list = U_list

        self.epsilon = float(epsilon)
        self.n_leapfrog = int(n_leapfrog)
        self.mass_a = float(mass_a)
        self.mass_z = float(mass_z)
        self.s_step = float(s_step)
        self.s_update_num = int(s_update_num)

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
        self.s = np.zeros((self.Lx, self.Ly), dtype=int)
        self._sync_U_from_a()

        self.boundary_mask_z = np.zeros((self.Lx, self.Ly), dtype=bool)
        self.boundary_mask_U = np.zeros((self.Lx, self.Ly, 2), dtype=bool)
        self._init_config()
        self._wrap_phase_inplace()
        self._sync_U_from_a()
        self._normalize_z_inplace()

        self.accepted_hmc = 0
        self.attempted_hmc = 0
        self.accepted_metro = 0
        self.attempted_metro = 0

    @property
    def accept_rate(self):
        '''acceptance rate of HMC and metro updates.'''
        if self.attempted_hmc == 0:
            hmc_rate = 0.0
        else:
            hmc_rate = self.accepted_hmc / self.attempted_hmc
        if self.attempted_metro == 0:
            metro_rate = 0.0
        else:
            metro_rate = self.accepted_metro / self.attempted_metro
        return hmc_rate, metro_rate

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
    # ---------- Forces / momenta with frozen boundary -----------
    # ------------------------------------------------------------

    def _z_force(self, mod=0):
        """OBC z-force, zeroed on the frozen boundary sites."""
        result = super()._z_force(mod=mod)
        result[self.boundary_mask_z] = 0.0
        return result

    def _a_force(self):
        """OBC gauge force -dS/da, zeroed on the frozen boundary links."""
        F = super()._a_force()
        F[self.boundary_mask_U] = 0.0
        return F

    def _sample_momenta(self):
        p_a, p_z = super()._sample_momenta()
        p_a[self.boundary_mask_U] = 0.0
        p_z[self.boundary_mask_z] = 0.0j
        p_z = self._project_tangent(self.z, p_z)
        return p_a, p_z

    def hmc_step(self, epsilon=None, n_leapfrog=None, mod=0):
        """One global HMC trajectory.

        Overridden only to guard the geodesic z-update against zero-norm
        momenta: frozen-boundary sites carry p_z == 0, so p_z_norm == 0 there
        and the base (PBC) hmc_step would emit NaN via p_z / p_z_norm.
        Otherwise identical to CPN_RefVil_HMCSampler.hmc_step: a is not wrapped
        during integration (the Villain plaquette force needs the continuous
        field), and a + s are wrapped/shifted once at the end.
        """
        eps = self.epsilon if epsilon is None else float(epsilon)
        nlf = self.n_leapfrog if n_leapfrog is None else int(n_leapfrog)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if nlf <= 0:
            raise ValueError("n_leapfrog must be positive")

        z_old = self.z.copy()
        a_old = self.a.copy()
        U_old = self.U.copy()
        s_old = self.s.copy()

        p_a, p_z = self._sample_momenta()
        H_old = self.hamiltonian(p_a, p_z, mod=mod)

        force_a = self._a_force()
        force_z = self._project_tangent(self.z, self._z_force(mod=mod))
        p_a += 0.5 * eps * force_a
        p_z += 0.5 * eps * force_z
        p_z = self._project_tangent(self.z, p_z)

        for step in range(nlf):
            # update a (no wrap here; see _apply_periodicity_wrap below)
            self.a += eps * p_a / self.mass_a
            self._sync_U_from_a()

            # update z on the |z|^2 = 1 sphere; guard p_z_norm == 0 sites
            p_z_norm = np.linalg.norm(p_z, axis=-1)
            safe_mask = p_z_norm > 1e-15
            p_z_unit = np.zeros_like(p_z)
            p_z_unit[safe_mask] = p_z[safe_mask] / p_z_norm[safe_mask, None]
            cos_p = np.cos(eps * p_z_norm / self.mass_z)
            sin_p = np.sin(eps * p_z_norm / self.mass_z)
            p_z = cos_p[:, :, None] * p_z - sin_p[:, :, None] * self.z * p_z_norm[:, :, None]
            self.z = cos_p[:, :, None] * self.z + sin_p[:, :, None] * p_z_unit
            self._normalize_z_inplace()
            p_z = self._project_tangent(self.z, p_z)

            # update momenta
            force_a = self._a_force()
            force_z = self._project_tangent(self.z, self._z_force(mod=mod))
            coeff = 1.0 if step < nlf - 1 else 0.5
            p_a += coeff * eps * force_a
            p_z += coeff * eps * force_z
            p_z = self._project_tangent(self.z, p_z)

        # wrap a into [-pi, pi] and shift s so da + 2*pi*s is invariant
        self._apply_periodicity_wrap()

        H_new = self.hamiltonian(p_a, p_z, mod=mod)
        dH = H_new - H_old

        self.attempted_hmc += 1
        if np.log(np.random.rand()) < -dH:
            self.accepted_hmc += 1
            return True, dH

        self.z = z_old
        self.a = a_old
        self.U = U_old
        self.s = s_old
        return False, dH

    # ------------------------------------------------------------
    # ---------- Measurements -------------------------------------
    # ------------------------------------------------------------

    # def get_vortex(self):
    #     '''
    #     Total integer winding through the single plaquette.
    #     Use 3 methods: U, z and s

    #     :return vortex: (3,) array of integer vortex from U, z and s
    #     '''
    #     L = self.L
    #     a = self.a
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
    #     vs = np.sum(self.s[:L, :L])
    #     return np.array([vU, vz, vs])

    def get_topo(self):
        '''
        Total topo charge through the single plaquette.
        Use 3 methods: U, z and s

        :return topo: (3,) array of integer topo charge from U, z and s
        '''
        L = self.L
        a = self.a
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
        qs = np.sum(temp_U[:L, :L] + self.s[:L, :L])
        return np.array([qU, qz, qs])

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
def run_RefVil_HMC(args):
    """
    Run one HMC of the RefVil 1-plaquette model and return the Berry-flux
    measurements together with the coarse boundary parameters.

    args = (N, L, beta, beta1, alpha, alpha1, boundary_BC, boundary_pad, mod,
            boundary_n_therm, s_step, s_update_num, n_therm, n_meas,
            meas_interval, epsilon, n_leapfrog, mass_a, mass_z)

    :return v_array:    measured vortex numbers, shape (n_meas, 3)
    :return da_U:       flux through the coarse plaquette from U, scalar
    :return da_z:       flux through the coarse plaquette from z, scalar
    :return acc_hmc:    HMC acceptance rate of the interior sampler (cumulative
                        over its thermalization + measurement sweeps)
    :return acc_metro:  s-Metropolis acceptance rate of the interior sampler
    """
    (N, L, beta, beta1, alpha, alpha1, boundary_BC, boundary_pad, mod,
     boundary_n_therm, s_step, s_update_num, n_therm, n_meas, meas_interval,
     epsilon, n_leapfrog, mass_a, mass_z) = args

    # ---------- boundary thermalization ----------
    if boundary_BC == 'OBC':
        sampler0 = CPN_RefVil_HMCSampler_OBC(
            N=N, Lx=(L + 1) + 2 * boundary_pad, Ly=(L + 1) + 2 * boundary_pad,
            beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
            epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z,
            s_step=s_step, s_update_num=s_update_num)
    elif boundary_BC == 'PBC':
        sampler0 = CPN_RefVil_HMCSampler(
            N=N, Lx=(L + 1) + 2 * boundary_pad - 1, Ly=(L + 1) + 2 * boundary_pad - 1,
            beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
            epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z,
            s_step=s_step, s_update_num=s_update_num)
    else:
        raise Exception("boundary_BC should be 'OBC' or 'PBC'")
    if boundary_pad < 0:
        raise Exception("boundary_pad should be non-negative")

    for _ in range(boundary_n_therm):
        sampler0.sweep(mod=mod)

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

    sampler = CPN_RefVil_HMCSampler_1plaq(
        N=N, L=L, beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
        z_list=z_list, U_list=U_list, epsilon=epsilon, n_leapfrog=n_leapfrog,
        mass_a=mass_a, mass_z=mass_z, s_step=s_step, s_update_num=s_update_num)
    a_list_U, a_list_z, _ = extract_param(L, z_list, U_list)
    da_U = np.sum(a_list_U)
    da_z = np.sum(a_list_z)

    # ---------- thermalize interior ----------
    for _ in range(n_therm):
        sampler.sweep(mod=mod)

    # ---------- measurements ----------
    # v_array = []
    q_array = []
    for i in range(n_meas):
        sampler.sweep(mod=mod)
        if i % meas_interval == 0:
            # v_array.append(sampler.get_vortex())
            q_array.append(sampler.get_topo())
    q_array = np.array(q_array)  # shape (n_meas, 3)
    v_array = q_array - np.array([da_U, da_z, da_U])[None, :] / 2 / np.pi

    acc_hmc, acc_metro = sampler.accept_rate
    return v_array, da_U, da_z, acc_hmc, acc_metro

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
    # beta1 = 1.0
    # alpha = 1.0
    # alpha1 = 1.0

    # z_list = rand_fine_z(L, N)
    # U_list = rand_fine_U(L)

    # sampler = CPN_RefVil_HMCSampler_1plaq(
    #     N=N, L=L, beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
    #     z_list=z_list, U_list=U_list, epsilon=0.02, n_leapfrog=50,
    #     s_step=0.5, s_update_num=1, seed=None)

    # # boundary round-trip check: extract_param must match the sampler's own coarse view
    # a_list_U, a_list_z, phi_list = extract_param(L, z_list, U_list)
    # # print("a_list_U - get_boundary_coarse_conn :", a_list_U - sampler.get_boundary_coarse_conn())
    # # print("a_list_z - get_boundary_coarse_conn_z:", a_list_z - sampler.get_boundary_coarse_conn_z())

    # # acc, dH = [], []
    # # for _ in range(10):
    # #     acc_t, dH_t = sampler.sweep(mod=1)
    # #     acc.append(acc_t)
    # #     dH.append(dH_t)
    # # print("acc:", acc)
    # # print("dH :", dH)
    # # print("flux sample:", sampler.get_conn(), " vortex Q:", sampler.get_vortex())
    # print(np.sum(np.angle(U_list)) / 2/ np.pi)
    # print(np.sum(a_list_U) / 2/ np.pi)
    args = (2, 4, 0.5, 0.0, 0.2, 0.2, 'OBC', 2, 1, 100, 0.5, 1, 100, 10, 1, 0.02, 50, 1.0, 1.0)
    v_array, da_U, da_z, acc_hmc, acc_metro = run_RefVil_HMC(args)
    print("v_array:", v_array)
    print("da_U:", da_U, "da_z:", da_z)
    print("acc_hmc:", acc_hmc, "acc_metro:", acc_metro)
    X, freq_v = get_X_freq(v_array, da_U, zero_pad=2)
    print("X:", X)
    print("freq_v:", freq_v)