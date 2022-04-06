import numpy as np
import matplotlib.pyplot as pl 
import scipy.stats as st
import theano.tensor as tt
import theano.tensor.slinalg as sl
from ipdb import set_trace as stop
from sklearn.neighbors import NearestNeighbors

def chol_invert(A):
    """
    Return the inverse of a symmetric matrix using the Cholesky decomposition. The log-determinant is
    also returned
    
    Args:
        A : (N,N) matrix
    
    Returns:
        AInv: matrix inverse
        logDeterminant: logarithm of the determinant of the matrix 
    """
    L = np.linalg.cholesky(A)
    LInv = np.linalg.inv(L)
    AInv = np.dot(LInv.T, LInv)
    logDeterminant = -2.0 * np.sum(np.log(np.diag(LInv)))   # Why the minus sign?
    return AInv, logDeterminant

def covariance(x, lambda_gp, sigma_gp):
    return sigma_gp * np.exp(-0.5 * lambda_gp * x**2)

N = 10
x = np.linspace(0,8,N)
mean = np.zeros((N))
K = covariance(x[None,:] - x[:,None], 1.0, 1.0)
x_test = np.ones((N))
print(st.multivariate_normal.logpdf(x_test, mean=mean, cov=K))

nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(np.atleast_2d(x).T)
distances, indices = nbrs.kneighbors(np.atleast_2d(x).T)

K_inv, logdet_K = chol_invert(K)
print(-0.5 * N * np.log(2.0*np.pi) - 0.5 * logdet_K - 0.5 * (x_test-mean).T @ K_inv @ (x_test-mean))

A = tt.as_tensor(np.zeros_like(K))
D_inv = tt.as_tensor(np.zeros_like(K))
I = tt.as_tensor(np.identity(N))

D_inv = tt.set_subtensor(D_inv[0,0], K[0,0])
for i in range(N-1):
    Pa = indices[i+1,:]
    Pa = Pa[Pa < i+1]
    Pa2 = np.atleast_2d(Pa).T
    A = tt.set_subtensor(A[i+1,Pa], sl.solve(K[Pa,Pa2], K[i+1,Pa]))
    D_inv = tt.set_subtensor(D_inv[i+1,i+1], 1.0/(K[i+1,i+1] - tt.dot(K[i+1,Pa], A[i+1,Pa])))

K_NNGP_inv = tt.dot(tt.dot(I - A.T, D_inv), I - A)
logdet_NNGP = np.sum(np.log(1.0/np.diag(D)))