import numpy as np
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


# Set matplotlib and seaborn plotting style ------------------------
sns.set_style("darkgrid")
np.random.seed(42)


# 1. synthetic data ---------------------------------------------------
# Define the true function that we want to regress on
f_sin = lambda x: (np.sin(x)).flatten()

n1 = 8  # Number of points to condition on (training points)
n2 = 75  # Number of points in posterior (test points)
ny = 5  # Number of functions that will be sampled from the posterior
domain = (-6, 6)

# Sample observations (X1, y1) on the function
X1 = np.random.uniform(domain[0]+2, domain[1]-2, size=(n1, 1))
y1 = f_sin(X1)
# Predict points at uniform spacing to capture function
X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)
# Compute posterior mean and covariance


# 2. Covariance functions --------------------------------------------
def exponentiated_quadratic01(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian) 
    #xa, xb = X1, X2
    sq_norm = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * (-0.5)
    return np.exp(sq_norm)

def exponentiated_quadratic02(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian) 
    #param01 = np.random.rand(xa.shape[0], xa.shape[1])
    param01 = np.repeat(0.5, xa.shape[0]).reshape(-1, 1) + np.random.rand(xa.shape[0], xa.shape[1])/2
    sq_norm = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * (-param01)
    return np.exp(sq_norm)


# 3. Display covariance -----------------------------------------------
nb_of_samples = 41  # Number of points in each function
Σ = exponentiated_quadratic01(X2, X2)  # Kernel of data points
viz_covariance(Σ)

# repeat number n times
Σ_non = exponentiated_quadratic02(X2, X2)  # Kernel of data points
viz_covariance(Σ_non)

# random number generator
def exponentiated_quadratic02(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian) 
    #param01 = np.random.rand(xa.shape[0], xa.shape[1])
    param01 = np.repeat(0.5, xa.shape[0]) + np.random.randint(0, 10, xa.shape[0])/100
    sq_norm = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') * (-param01.reshape(-1, 1))
    return np.exp(sq_norm)

_, Σ2 = GP(X1, y1, X2, exponentiated_quadratic02)
viz_covariance(Σ2)
