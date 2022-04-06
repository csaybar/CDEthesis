from matplotlib import pyplot as plt
import torch
import math

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


## Create  synthetic data -------------------------
# Training data is 20 points in [0,1] inclusive regularly spaced
train_x_mean = torch.linspace(0, 1, 20)
train_x_stdv = torch.linspace(0.03, 0.01, 20) # assume the variance shrinks the closer we get to 1
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x_mean * (2 * math.pi)) + torch.randn(train_x_mean.size()) * 0.2

f, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.errorbar(train_x_mean, train_y, xerr=(train_x_stdv * 2), fmt="k*", label="Train Data")
ax.legend()
plt.show()


# Create a variational/approximate GP models ---------------------------------------------------
# 1. A GP Model (gpytorch.models.ApproximateGP) - This handles basic variational inference.
# 2. A Variational distribution (gpytorch.variational._VariationalDistribution) - This tells us what 
# form the variational distribution q(u) should take.
# 3. A Variational strategy (gpytorch.variational._VariationalStrategy) - This tells us how to transform 
# a distribution q(u) over the inducing point values to a distribution q(f) over the latent function values
# for some input x.

# Here, we use a VariationalStrategy with learn_inducing_points=True, 
# and a CholeskyVariationalDistribution. These are the most straightforward and 
# common options.
class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
inducing_points = torch.randn(10, 1)
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Training the model with uncertain features -----------------------------------------------
# The cell below trains the model above, learning both the hyperparameters of the Gaussian 
# process and the parameters of the neural network in an end-to-end fashion using Type-II MLE.

# Unlike when using the exact GP marginal log likelihood, performing variational inference allows
# us to make use of stochastic optimization techniques. For this example, we'll do one epoch of 
# training. Given the small size of the neural network relative to the size of the dataset, this 
# should be sufficient to achieve comparable accuracy to what was observed in the DKL paper.

# The optimization loop differs from the one seen in our more simple tutorials in that it involves
# looping over both a number of training iterations (epochs) and minibatches of the data. However, 
# the basic process is the same: for each minibatch, we forward through the model, compute the 
# loss (the VariationalELBO or ELBO), call backwards, and do a step of optimization.
training_iter = 400
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
iterator = range(training_iter)
for i in iterator:
    # First thing: draw a sample set of features from our distribution
    train_x_sample = torch.distributions.Normal(train_x_mean, train_x_stdv).rsample()
    
    # Now do the rest of the training loop
    optimizer.zero_grad()
    output = model(train_x_sample)
    loss = -mll(output, train_y)    
    loss.backward()
    optimizer.step()
    

# Making Predictions
# The next cell gets the predictive covariance for the test set (and also technically 
# gets the predictive mean, stored in preds.mean()). Because the test set is substantially
# smaller than the training set, we don't need to make predictions in mini batches here, 
# although this can be done by passing in minibatches of test_x rather than the full tensor.

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.errorbar(train_x_mean.numpy(), train_y.numpy(), xerr=train_x_stdv, fmt='k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()