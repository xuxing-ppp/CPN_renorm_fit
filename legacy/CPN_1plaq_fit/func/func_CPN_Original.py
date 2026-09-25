import numpy as np
from numpy.random import rand, randn, vonmises
from tqdm import tqdm

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
    return [r + 1j * i for (r,i) in phi0]
    
class CPNSampler:
    """
    Clean CP^{N-1} Over-Heatbath Monte-Carlo sampler.

    System size: Lx * Ly

    N-dim normalized complex vector on sites, and U(1) phases on links.

    Action is "half-Villainized" i.e. with cos(da) term but no integer variable.

    U_init is the initial value of every Ux and Uy (given by phase).
    """

    def __init__(self, N, Lx, Ly, beta, alpha1, U_init_phase=None, seed=None):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        self.beta = beta
        self.alpha1 = alpha1
        self.V = Lx * Ly

        if seed is not None:
            np.random.seed(seed)

        # z-field: shape (Lx,Ly,N) complex
        self.z = self._random_unit_vectors((Lx, Ly, N))

        # U(1) link field: shape (Lx,Ly,2) complex of unit norm
        # dir=0 → x direction, dir=1 → y direction
        if U_init_phase is None:
            phases = 2*np.pi * rand(Lx, Ly, 2)
            self.U = np.exp(1j * phases)
        else:
            phases = U_init_phase * np.ones((Lx, Ly, 2))
            self.U = np.exp(1j * phases)

    # ------------------------------------------------------------
    # ---------- Utility: Random unit vectors ---------------------
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

    # ------------------------------------------------------------
    # ---------- Force terms --------------------------------------
    # ------------------------------------------------------------

    def force_z(self, x, y):
        """
        Compute local force F_z(x, y) = 2N*beta sum_mu lambda*(x,mu) z(x+mu) + lambda(x-mu,mu) z(x-mu)
        Returns shape (N,) complex.
        """

        Lx, Ly = self.Lx, self.Ly
        N = self.N

        # neighbors with periodic bc
        xp = (x + 1) % Lx
        xm = (x - 1) % Lx
        yp = (y + 1) % Ly
        ym = (y - 1) % Ly

        z = self.z
        U = self.U

        if Lx > 1 and Ly > 1:
            Fx = U[xm, y, 0] * z[xm, y] + np.conj(U[x, y, 0]) * z[xp, y]
            Fy = U[x, ym, 1] * z[x, ym] + np.conj(U[x, y, 1]) * z[x, yp]
        elif Ly == 1 and Lx > 1:
            Fx = U[xm, y, 0] * z[xm, y] + np.conj(U[x, y, 0]) * z[xp, y]
            Fy = 0
        elif Lx == 1 and Ly > 1:
            Fx = 0
            Fy = U[x, ym, 1] * z[x, ym] + np.conj(U[x, y, 1]) * z[x, yp]
        elif Lx == Ly == 1:
            Fx = np.zeros(N, dtype=complex)
            Fy = np.zeros(N, dtype=complex)

        return 2 * self.N * self.beta * (Fx + Fy)

    def force_U(self, x, y, mu):
        """
        Force for U(1) link:
        F = 2N*beta z(x+mu) dot conj(z(x)) + alpha1 * (plaq terms)
        return a complex number
        """
        Lx, Ly = self.Lx, self.Ly
        # neighbors with periodic bc
        xp = (x + 1) % Lx
        xm = (x - 1) % Lx
        yp = (y + 1) % Ly
        ym = (y - 1) % Ly

        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        alpha1 = self.alpha1

        if mu == 0:
            if Ly > 1:
                return 2 * N * beta * np.vdot(z[x, y], z[xp, y]) + alpha1 * (np.conj(U[xp, y, 1]) * U[x, yp, 0] * U[x, y, 1] + U[xp, ym, 1] * U[x, ym, 0] * np.conj(U[x, ym, 1]))
            elif Ly == 1:
                return 2 * N * beta * np.vdot(z[x, y], z[xp, y])
        elif mu == 1:
            if Lx > 1:
                return 2 * N * beta * np.vdot(z[x, y], z[x, yp]) + alpha1 * (np.conj(U[x, yp, 0]) * U[xp, y, 1] * U[x, y, 0] + U[xm, yp, 0] * U[xm, y, 1] * np.conj(U[xm, y, 0]))
            elif Lx == 1:
                return 2 * N * beta * np.vdot(z[x, y], z[x, yp])
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
        where F = < z(x+mu), z(x) > + plaq term
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
        if Fnorm < 1e-10:
            # no meaningful force → random unit vector
            self.z[x, y] = self._random_unit_vectors((1, self.N))[0]
            return
        z = self.z[x, y]
        z_new = 2 * np.real(np.vdot(z, F)) / Fnorm**2 * F - z
        self.z[x, y] = z_new / np.linalg.norm(z_new)

    def or_update_U(self, x, y, mu):
        F = self.force_U(x, y, mu)
        Fnorm = np.linalg.norm(F)
        if Fnorm < 1e-10:
            self.U[x, y, mu] = np.exp(1j * 2*np.pi * rand())
            return
        U = self.U[x, y, mu]
        U_new = 2 * np.real(np.conj(U) * F) / Fnorm**2 * F - U
        self.U[x, y, mu] = U_new / np.abs(U_new)

    # ------------------------------------------------------------
    # ---------- MC Sweep -----------------------------------------
    # ------------------------------------------------------------

    def sweep(self, heatbath_fraction=0.5):
        """Heatbath + overrelaxation sweep."""
        # mu = -1 stands for z update
        indices = [(x, y, mu) for y in range(self.Ly) for x in range(self.Lx) for mu in (-1, 0, 1)]
        np.random.shuffle(indices)
        for (x, y, mu) in indices:
            if mu == -1:
                if rand() < heatbath_fraction:
                    self.hb_update_z(x, y)
                else:
                    self.or_update_z(x, y)
            else:
                if rand() < heatbath_fraction:
                    self.hb_update_U(x, y, mu)
                else:
                    self.or_update_U(x, y, mu)

class CPNSampler_OBC:
    """
    Clean CP^{N-1} Over-Heatbath Monte-Carlo sampler, OBC, boundary not fixed.

    System size: Lx * Ly

    N-dim normalized complex vector on sites, and U(1) phases on links.

    Action is "half-Villainized" i.e. with cos(da) term but no integer variable.

    U_init is the initial value of every Ux and Uy (given by phase).
    """

    def __init__(self, N, Lx, Ly, beta, alpha1, U_init_phase=None, seed=None):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        self.beta = beta
        self.alpha1 = alpha1

        if seed is not None:
            np.random.seed(seed)

        # z-field: shape (Lx,Ly,N) complex
        self.z = self._random_unit_vectors((Lx, Ly, N))

        # U(1) link field: shape (Lx,Ly,2) complex of unit norm
        # dir=0 → x direction, dir=1 → y direction
        if U_init_phase is None:
            phases = 2*np.pi * rand(Lx, Ly, 2)
            self.U = np.exp(1j * phases)
        else:
            phases = U_init_phase * np.ones((Lx, Ly, 2))
            self.U = np.exp(1j * phases)

    # ------------------------------------------------------------
    # ---------- Utility: Random unit vectors ---------------------
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

    # ------------------------------------------------------------
    # ---------- Force terms --------------------------------------
    # ------------------------------------------------------------

    def force_z(self, x, y):
        """
        Compute local force F_z(x, y) = 2N*beta sum_mu lambda*(x,mu) z(x+mu) + lambda(x-mu,mu) z(x-mu)
        Returns shape (N,) complex.
        """

        Lx, Ly = self.Lx, self.Ly

        # neighbors with OBC
        xp = (x + 1)
        xm = (x - 1)
        yp = (y + 1)
        ym = (y - 1)

        z = self.z
        U = self.U

        F = 0
        if x > 0:
            F += U[xm, y, 0] * z[xm, y]
        if x < Lx - 1:
            F += np.conj(U[x, y, 0]) * z[xp, y]
        if y > 0:
            F += U[x, ym, 1] * z[x, ym]
        if y < Ly - 1:
            F += np.conj(U[x, y, 1]) * z[x, yp]

        return 2 * self.N * self.beta * F

    def force_U(self, x, y, mu):
        """
        Force for U(1) link:
        F = 2N*beta z(x+mu) dot conj(z(x)) + alpha1 * (plaq terms)
        return a complex number
        """
        Lx, Ly = self.Lx, self.Ly
        # neighbors with OBC
        xp = (x + 1)
        xm = (x - 1)
        yp = (y + 1)
        ym = (y - 1)

        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        alpha1 = self.alpha1

        F = 0
        if mu == 0:
            if x == Lx - 1:
                raise Exception("index out of range")
            if y > 0:
                F += U[xp, ym, 1] * U[x, ym, 0] * np.conj(U[x, ym, 1]) * alpha1
            if y < Ly - 1:
                F += np.conj(U[xp, y, 1]) * U[x, yp, 0] * U[x, y, 1] * alpha1
            return 2 * N * beta * np.vdot(z[x, y], z[xp, y]) + F
        elif mu == 1:
            if y == Ly - 1:
                raise Exception("index out of range")
            if x > 0:
                F += U[xm, yp, 0] * U[xm, y, 1] * np.conj(U[xm, y, 0]) * alpha1
            if x < Lx - 1:
                F += np.conj(U[x, yp, 0]) * U[xp, y, 1] * U[x, y, 0] * alpha1
            return 2 * N * beta * np.vdot(z[x, y], z[x, yp]) + F
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
        where F = < z(x+mu), z(x) > + plaq term
        """
        if (x == self.Lx - 1 and mu ==0) or (y == self.Ly - 1 and mu == 1):
            return
        
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
        if (x == self.Lx - 1 and mu ==0) or (y == self.Ly - 1 and mu == 1):
            return
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
        indices = [(x, y, mu) for y in range(self.Ly) for x in range(self.Lx) for mu in (-1, 0, 1)]
        np.random.shuffle(indices)
        for (x, y, mu) in indices:
            if mu == -1:
                if rand() < heatbath_fraction:
                    self.hb_update_z(x, y)
                else:
                    self.or_update_z(x, y)
            else:
                if rand() < heatbath_fraction:
                    self.hb_update_U(x, y, mu)
                else:
                    self.or_update_U(x, y, mu)