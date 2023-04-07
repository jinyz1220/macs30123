from numba import jit

@jit('f8[:,:](f4, f4, f8[:,:], f8[:,:])', nopython=True) # output(input), datatype
def simulate_path(rho, mu, eps_mat, z_mat):
    T, S = eps_mat.shape
    for s_ind in range(S):
        z_tm1 = mu
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat
