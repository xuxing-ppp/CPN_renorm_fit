import numpy as np
from numpy.fft import fftn, ifftn
import scipy 


def vdot_z(z1, z2):
    return np.einsum("ijk,ijk->ij", np.conj(z1), z2)

def vouter_z(z1, z2):
    return np.einsum("ijk,ijl->ijkl", z1, np.conj(z2))

class CPN_halfRefVil_HMCSampler:
    """
    CP^{N-1} + U(1) sampler with global constrained HMC updates.

    - z field: shape (Lx, Ly, N), complex, with per-site norm |z|^2 = 1.
    - U field: shape (Lx, Ly, 2), complex phases on links.
    - a field: real phases such that U = exp(1j * a).

    Action is half-Villainized improved form with beta1 * |z' z|^2 term or log I0(beta1 * |z' z|) term,
    determined by mod = 0, 1
    """

    def __init__(
        self,
        Lx,
        Ly,
        N,
        beta,
        beta1,
        alpha,
        seed=None,
        epsilon=0.05,
        n_leapfrog=20,
        mass_a=1.0,
        mass_z=1.0,
    ):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        self.beta = beta
        self.beta1 = beta1
        self.alpha = alpha
        self.V = Lx * Ly

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

        self.z = self._random_unit_vectors((Lx, Ly, N))

        self.a = 2 * np.pi * np.random.rand(Lx, Ly, 2)
        self._wrap_phase_inplace()
        self._sync_U_from_a()

        self.accepted = 0
        self.attempted = 0

    @property
    def accept_rate(self):
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted

    def _random_unit_vectors(self, shape):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
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
        """
        Project complex field onto tangent space of |z|^2=1 manifold.
        Constraint is Re(<z, field>) = 0 at each site.
        """
        overlap = np.einsum("ijk,ijk->ij", np.conj(z), field)
        return field - np.real(overlap)[:, :, None] * z

    def _spin_inner_x(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=0))

    def _spin_inner_y(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=1))

    def _plaquette(self):
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]
        return Ux * np.roll(Uy, -1, axis=0) * np.conj(np.roll(Ux, -1, axis=1) * Uy)

    def action_density(self, mod=0):
        z = self.z
        U = self.U
        V = self.V

        cp_term = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term += np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        if mod == 0:
            cp_term_1 = np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) ** 2 - 1
            cp_term_1 += np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) ** 2 - 1
            cp_term_1 *= self.N * self.beta1
        elif mod == 1:
            cp_term_1 = np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) * 2 * self.N * self.beta1))
            cp_term_1 += np.sign(self.beta1) * np.log(scipy.special.i0(np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) * 2 * self.N * self.beta1))
        plaq_term = np.real(self._plaquette())

        return -2.0 * self.N * self.beta * (cp_term - 2) - cp_term_1 - self.alpha * (plaq_term - 1)

    def action(self, mod=0):
        return np.sum(self.action_density(mod=mod))

    def _z_force(self, mod=0):
        """
        Returns force F_z = -dS/d(conj(z)) * 2.
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
            F += np.roll(U[:, :, 0], 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0)
            F += np.conj(U[:, :, 0])[:, :, None] * np.roll(z, -1, axis=0)
            if mod == 0:
                F1 += np.roll(inner_x, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * beta1
                F1 += np.conj(inner_x)[:, :, None] * np.roll(z, -1, axis=0) * beta1
            elif mod == 1:
                inner_x_abs = np.abs(inner_x)
                temp = scipy.special.i1(inner_x_abs * 2 * N * beta1) / scipy.special.i0(inner_x_abs * 2 * N * beta1) * inner_x / inner_x_abs
                F1 += np.roll(temp, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=0) * np.abs(beta1)

        if self.Ly > 1:
            F += np.roll(U[:, :, 1], 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1)
            F += np.conj(U[:, :, 1])[:, :, None] * np.roll(z, -1, axis=1)
            if mod == 0:
                F1 += np.roll(inner_y, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * beta1
                F1 += np.conj(inner_y)[:, :, None] * np.roll(z, -1, axis=1) * beta1
            elif mod == 1:
                inner_y_abs = np.abs(inner_y)
                temp = scipy.special.i1(inner_y_abs * 2 * N * beta1) / scipy.special.i0(inner_y_abs * 2 * N * beta1) * inner_y / inner_y_abs
                F1 += np.roll(temp, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=1) * np.abs(beta1)

        return N * beta * F * 2.0 + N * F1 * 2.0

    def _link_force_complex(self):
        """
        Returns complex F for each link such that local action is
        S_link = -Re(conj(U) * F).
        """
        z = self.z
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]

        F = np.zeros((self.Lx, self.Ly, 2), dtype=complex)

        F[:, :, 0] += 2.0 * self.N * self.beta * self._spin_inner_x()
        F[:, :, 1] += 2.0 * self.N * self.beta * self._spin_inner_y()

        if self.alpha != 0.0:
            if self.Ly > 1:
                staple_x_1 = np.conj(np.roll(Uy, -1, axis=0)) * np.roll(Ux, -1, axis=1) * Uy
                staple_x_2 = np.roll(Uy, (-1, 1), axis=(0, 1)) * np.roll(Ux, 1, axis=1) * np.conj(np.roll(Uy, 1, axis=1))
                F[:, :, 0] += self.alpha * (staple_x_1 + staple_x_2)
            if self.Lx > 1:
                staple_y_1 = np.conj(np.roll(Ux, -1, axis=1)) * np.roll(Uy, -1, axis=0) * Ux
                staple_y_2 = np.roll(Ux, (1, -1), axis=(0, 1)) * np.roll(Uy, 1, axis=0) * np.conj(np.roll(Ux, 1, axis=0))
                F[:, :, 1] += self.alpha * (staple_y_1 + staple_y_2)

        return F

    def _a_force(self):
        """
        Returns force for momentum equation da/dt = p_a/mass_a,
        dp_a/dt = force_a = -dS/da.
        """
        F_complex = self._link_force_complex()
        return np.imag(np.conj(self.U) * F_complex)

    def _sample_momenta(self):
        p_a = np.sqrt(self.mass_a) * np.random.randn(self.Lx, self.Ly, 2)

        p_z = np.sqrt(self.mass_z) * (
            np.random.randn(self.Lx, self.Ly, self.N)
            + 1j * np.random.randn(self.Lx, self.Ly, self.N)
        )
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

            # rot_mat = scipy.linalg.expm(eps * (vouter_z(p_z, self.z) - vouter_z(self.z, p_z)) / self.mass_z) * np.exp(eps * vdot_z(p_z, self.z) / self.mass_z)[:, :, None, None]
            # self.z = np.einsum("ijkl,ijl->ijk", rot_mat, self.z)
            # p_z = np.einsum("ijkl,ijl->ijk", rot_mat, p_z)
            p_z_norm = np.linalg.norm(p_z, axis=-1)
            p_z_unit = p_z / p_z_norm[:, :, None]
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
    
    # ----------------------------------- Observables -----------------------------------
    
    def argzz_loop_xy_list(self, xy_list):
        '''
        calculate cos(arg(ZZ) + arg(ZZ) + arg(ZZ) + arg(ZZ)), and return (len(xy_list),) array.
        '''
        z = self.z
        w_list = []
        for x, y in xy_list:
            temp = vdot_z(z, np.roll(z, -x, 0)) * vdot_z(np.roll(z, -x, 0), np.roll(z, (-x,-y), (0,1))) * vdot_z(np.roll(z, (-x,-y), (0,1)), np.roll(z, -y, 1)) * vdot_z(np.roll(z, -y, 1), z)
            w_z_renorm = np.cos(np.angle(temp))
            w_list.append(np.mean(w_z_renorm))

        return np.asarray(w_list)
    
    def wilson_loop_xy_list(self, xy_list):
        '''
        Use two methods: Re(U U U U) and Re(Uz Uz Uz Uz) and return (len(xy_list), 2) array.
        '''
        U = self.U
        z = self.z
        U11 = U[:,:,0] * np.roll(U, -1, axis=0)[:,:,1] * np.conj(np.roll(U, -1, axis=1)[:,:,0] * U[:,:,1])
        U_z = np.zeros_like(U)
        temp = vdot_z(z, np.roll(z, -1, 0))
        U_z[:,:,0] = temp / np.abs(temp)
        temp = vdot_z(z, np.roll(z, -1, 1))
        U_z[:,:,1] = temp / np.abs(temp)
        U_z11 = U_z[:,:,0] * np.roll(U_z, -1, axis=0)[:,:,1] * np.conj(np.roll(U_z, -1, axis=1)[:,:,0] * U_z[:,:,1])

        w_list = []
        for x, y in xy_list:
            Uxy = np.prod([np.roll(U11, (-i, -j), axis=(0, 1)) for i in range(x) for j in range(y)], axis=0)
            w_U = np.real(Uxy)
            U_zxy = np.prod([np.roll(U_z11, (-i, -j), axis=(0, 1)) for i in range(x) for j in range(y)], axis=0)
            w_z = np.real(U_zxy)
            w_list.append([np.mean(w_U), np.mean(w_z)])

        return np.asarray(w_list)

    def P_exp(self):
        """
        Expectation value of P = z bar(z).
        """
        P = np.einsum("ijk,ijl->ijkl", self.z, np.conj(self.z))
        return np.mean(P, axis=(0, 1))

    def PP_corr_k(self):
        """
        Return fourier transformation of connected PP correlation.
        """
        N = self.N
        P = np.einsum("ijk,ijl->ijkl", self.z, np.conj(self.z))
        Px = P - 1 / N * np.eye(N)[None, None, :, :]
        Pk = fftn(Px, axes=(0, 1))
        S = np.sum(np.abs(Pk) ** 2, axis=(2, 3)) / self.V
        return S

    def conn_PP_corr(self):
        """Return real-space connected PP correlation, shape (Lx, Ly)."""
        return ifftn(self.PP_corr_k()).real

    def PP_corr_xy(self, x, y):
        """
        Calculate PP correlation between (0,0) and (x,y).
        """
        corr = self.conn_PP_corr() + 1 / self.N
        return corr[x % self.Lx, y % self.Ly]

    def PP_corr_coord(self, coord):
        """
        Calculate PP correlation for coordinate list [[x1, y1], [x2, y2], ...].
        """
        corr = self.conn_PP_corr() + 1 / self.N
        return np.asarray([corr[x % self.Lx, y % self.Ly] for x, y in coord])

    def poly_loop(self):
        """
        Average Polyakov loop along y direction using U and z.

        Returns two complex numbers.
        """
        poly_arr = np.prod(self.U[:, :, 1], axis=1)
        z = self.z
        temp = np.einsum("ijk,ijk->ij", np.conj(z), np.roll(z, -1, 1))
        temp = temp / np.abs(temp)
        poly_arr_z = np.prod(temp, axis=1)
        return np.mean(poly_arr), np.mean(poly_arr_z)

    def topo_charge(self):
        """
        Skyrmion number of the system.

        Use two methods: arg(UUUU) and (arg(ZZ) + arg(ZZ) + ...), and return two float.
        """
        U = self.U
        z = self.z

        Q_U = np.angle(
            U[:, :, 0]
            * np.roll(U, -1, axis=0)[:, :, 1]
            * np.conj(np.roll(U, -1, axis=1)[:, :, 0] * U[:, :, 1])
        )
        temp0 = (
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), z)
        )
        temp1 = (
            vdot_z(z, np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        )
        Q_z = np.angle(temp0) + np.angle(temp1)

        return np.sum(Q_U) / (2 * np.pi), np.sum(Q_z) / (2 * np.pi)

if __name__ == "__main__":
    Lx, Ly = 1, 8
    N = 2
    beta = 1.0
    beta1 = 1.0
    alpha = 1.0

    sampler = CPN_halfRefVil_HMCSampler(Lx, Ly, N, beta, beta1, alpha, epsilon=0.02, n_leapfrog=50, seed=123)

    acc, dH = [], []
    for _ in range(10):
        acc_t, dH_t = sampler.sweep()
        acc.append(acc_t)
        dH.append(dH_t)
    
    print(acc)
    print(dH)