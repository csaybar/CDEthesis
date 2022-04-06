import numpy as np
import gpytorch
import torch
        
class GridGPRegressionModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(GridGPRegressionModel, self).__init__(train_x, train_y, likelihood)
    num_dims = train_x.size(-1)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
  
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(train, val, epochs=10, lr=0.1, patience=10, save_best_model = "./bestmodel.pt", verbose = True):
  # Load train dataset
  train_x, train_y = train.values()
  train_x = torch.Tensor(train_x.to_numpy())
  train_x = train_x.cuda()
  train_y = torch.Tensor(np.array(train_y))
  train_y = train_y.cuda()
  
  # Load validation dataset
  val_x, val_y = val.values()
  val_x = torch.Tensor(val_x.to_numpy())
  val_x = val_x.cuda()
  val_y = torch.Tensor(np.array(val_y))
  val_y = val_y.cuda()
  
  # Set up the GP
  likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
  model = GridGPRegressionModel(train_x, train_y, likelihood).cuda()
    
  # Define the EarlyStopping callback
  early_stopping = EarlyStopping(patience=patience, verbose=verbose, path = save_best_model) 
  
  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Includes GaussianLikelihood parameters

  # "Loss" for GPs - the marginal log likelihood
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
  
  loss_train_container = []
  loss_val_container = []
  for i in range(epochs):
    # Preparing model for training
    model.train()
    likelihood.train()
    
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc train loss
    train_loss = -mll(output, train_y)
    loss_train_container.append(float(train_loss.cpu().detach().numpy()))
    
    # Calc val loss 
    with torch.no_grad():
        # Preparing model for testing
        model.eval()
        likelihood.eval()
        
        # Val loss
        val_output = model(val_x)
        val_loss = -mll(val_output, val_y)
        loss_val_container.append(float(val_loss.cpu().detach().numpy()))
        
        print('Iter %d/%d - train_Loss: %.3f  val_loss: %.3f' % (
          i + 1, epochs, train_loss.item(), val_loss.item()
        ), flush = True)
        try:
          early_stopping(val_loss, model)
        except:
          break
    # backprop gradients
    train_loss.backward()
    optimizer.step()
  return {'train': loss_train_container, 'val': loss_val_container}


def predict_gp(model, train, test):
    # Load weights
    state_dict = torch.load(model)
    
    # Load train dataset
    train_x, train_y = train.values()
    train_x = torch.Tensor(train_x.to_numpy())
    train_x = train_x.cuda()
    train_y = torch.Tensor(np.array(train_y))
    train_y = train_y.cuda()
    
    # Load test dataset
    test_x, test_y = test.values()
    test_x = torch.Tensor(test_x.to_numpy())
    test_x = test_x.cuda()
    test_y = torch.Tensor(np.array(test_y))
    test_y = test_y.cuda()

    # Set up the GP
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = GridGPRegressionModel(train_x, train_y, likelihood).cuda()    
    model.load_state_dict(state_dict)
    
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))    
    mean_values = observed_pred.mean.view(test_x.shape[0],)
    covariance_matrix = observed_pred.covariance_matrix
    return {'mean_values': mean_values.cpu().detach().numpy(), 'covariance_matrix': covariance_matrix.cpu().detach().numpy()}
