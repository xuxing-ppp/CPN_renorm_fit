import numpy as np
from numpy.random import rand, randn, vonmises
from numpy.fft import fftn, ifftn

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
    acf = acf.real                      # autocorrelation is real even for complex x
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

class CPNSampler:
    """
    Clean CP^{N-1} Over-Heatbath Monte-Carlo sampler.

    System size: Lx * Ly

    N-dim normalized complex vector on sites, and U(1) phases on links.

    Action is "half-Villainized" i.e. with cos(da) term but no integer variable.

    U_init is the initial value of every Ux and Uy (given by phase).
    """

    def __init__(self, Lx, Ly, N, beta, alpha, U_init_phase=None, seed=None):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        self.beta = beta
        self.alpha = alpha
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
        F = 2N*beta z(x+mu) dot conj(z(x)) + alpha * (plaq terms)
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
        alpha = self.alpha

        if mu == 0:
            if Ly > 1:
                return 2 * N * beta * np.vdot(z[x, y], z[xp, y]) + alpha * (np.conj(U[xp, y, 1]) * U[x, yp, 0] * U[x, y, 1] + U[xp, ym, 1] * U[x, ym, 0] * np.conj(U[x, ym, 1]))
            elif Ly == 1:
                return 2 * N * beta * np.vdot(z[x, y], z[xp, y])
        elif mu == 1:
            if Lx > 1:
                return 2 * N * beta * np.vdot(z[x, y], z[x, yp]) + alpha * (np.conj(U[x, yp, 0]) * U[xp, y, 1] * U[x, y, 0] + U[xm, yp, 0] * U[xm, y, 1] * np.conj(U[xm, y, 0]))
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

    # ------------------------------------------------------------
    # ---------- Measurements -------------------------------------
    # ------------------------------------------------------------ 
    
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
        '''
        Expectation value of P = z bar(z)
        '''
        P = np.einsum('ijk,ijl->ijkl', self.z, np.conj(self.z))
        return np.mean(P, axis=(0, 1))
    
    def PP_corr_k(self):
        '''
        Return fourier transformation of connected PP correlation
        '''
        N = self.N
        P = np.einsum('ijk,ijl->ijkl', self.z, np.conj(self.z))
        Px = P - 1 / N * np.eye(N)[None, None, :, :]
        Pk = fftn(Px, axes=(0, 1))
        S = np.sum(np.abs(Pk)**2, axis=(2, 3)) / self.V
        return S
    
    def conn_PP_corr(self):
        '''Return the real space connected PP correlation, shape (Lx, Ly) array'''
        return ifftn(self.PP_corr_k()).real
    
    def PP_corr_xy(self, x, y):
        '''
        Calculate the PP correlation between (0,0) and (x,y) (P = z bar(z), tr(Pi Pj) = |bar(zi) zj|^2)

        Note there's translation invariance.
        '''
        corr = self.conn_PP_corr() + 1 / self.N
        return corr[x % self.Lx, y % self.Ly]

    def PP_corr_coord(self, coord):
        '''
        Calculate the PP correlation between (0,0) and (x,y) (P = z bar(z), tr(Pi Pj) = |bar(zi) zj|^2)
        coord = [[x1, y1], [x2, y2], ...]
        '''
        corr = self.conn_PP_corr() + 1 / self.N
        return np.asarray([corr[x % self.Lx, y % self.Ly] for x, y in coord])
  
    def poly_loop(self):
        '''
        Average of polyakov loop along y direction, use U and z

        return two complex numbers
        '''
        poly_arr = np.prod(self.U[:, :, 1], axis=1)
        z = self.z
        temp = np.einsum('ijk,ijk->ij', np.conj(z), np.roll(z, -1, 1))
        temp = temp / np.abs(temp)
        poly_arr_z  = np.prod(temp, axis=1)
        return np.mean(poly_arr), np.mean(poly_arr_z)
    
    def topo_charge(self):
        '''
        The skyrmion number of the system.

        Use two methods: arg(UUUU) and (arg(ZZ) + arg(ZZ) + ...), and return two float.
        '''
        U = self.U
        z = self.z

        Q_U = np.angle(U[:,:,0] * np.roll(U, -1, axis=0)[:,:,1] * np.conj(np.roll(U, -1, axis=1)[:,:,0] * U[:,:,1]))
        temp0 = vdot_z(z, np.roll(z, -1, 0)) * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0,1))) * vdot_z(np.roll(z, -1, (0,1)), z)
        temp1 = vdot_z(z, np.roll(z, -1, (0,1))) * vdot_z(np.roll(z, -1, (0,1)), np.roll(z, -1, 1)) * vdot_z(np.roll(z, -1, 1), z)
        Q_z = np.angle(temp0) + np.angle(temp1)

        return np.sum(Q_U) / 2 / np.pi, np.sum(Q_z) / 2 / np.pi

if __name__ == "__main__":
    L = 20
    N = 10
    beta = 0.1
    alpha = 20
    x0 = 2
    y0 = 0

    sampler = CPNSampler(L, L, N, beta, alpha)
    print(sampler.wilson_loop11())
    print(sampler.wilson_loop_xy_list([[2,2]]))