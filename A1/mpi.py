import numpy as np
import time
from mpi4py import MPI
import scipy.stats as sts
import path

def sim_parallel(rho, mu, sigma, S, T):
  np.random.seed(25)
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Evenly distribute number of simulation runs across processes
  N = int(S / size)

  # set random seed based on rank
  np.random.seed(rank)

  # draw all idiosyncratic random shocks for this rank
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
  z_mat = np.zeros((T, N))

  # simulate life paths
  start_time = time.time()
  sub_z_mat = path.simulate_path(rho, mu, eps_mat, z_mat)

  # collect all z matrices from different cores
  z_mat_all = None
  if rank == 0:
      z_mat_all = np.empty((T, N*size))

  comm.Gather(sub_z_mat, z_mat_all, root=0)
  end_time = time.time()

  # print time elapsed
  if rank == 0:
    elapsed = end_time - start_time
    print("Elapsed time:", elapsed)

  return 

sim_parallel(0.5, 3.0, 1.0, 1000, int(4160))

