import scipy
from matplotlib import cm
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))

# Gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1), 
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12).T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance


# Illustrate covariance matrix and function
def viz_covariance(covariance):
    plt.imshow(covariance, cmap=cm.YlGnBu)
    plt.show()