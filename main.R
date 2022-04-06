#' Bayesian Non-Stationary Scalable Gaussian Process Model - main
#' @author Cesar Aybar and Francois Septier

library(sf)
library(sp)
library(stars)
library(gstat)
library(dplyr)
library(giscoR)
library(raster)
library(mapview)
library(Metrics)
library(reticulate)

source("utils.R")
source_python("main.py")

# 1. Create a world map ---------------------------------------------------
coordinates <- world_map_creator(2)
# mapview(coordinates %>% st_as_sf(coords = c("X", "Y"), crs = 4326), legend = FALSE)

# 2. synthetic data models ------------------------------------------------
ground_truth <- toy_dataset(coordinates = coordinates, type = "toy02")
sf_coordinates <- as.data.frame(ground_truth, xy = TRUE) %>% st_as_sf(coords = c("x", "y"), crs = 4326)
sf_coordinates <- na.omit(sf_coordinates)
# plot(ground_truth)


# 3. Sample data points ---------------------------------------------------
samples <- block_sampling(ground_truth = ground_truth, size = 2000, block = 1)
# plot(samples$train)
sample_train <- samples$train
sample_test <- samples$test
sample_val <- samples$val
# mapview(samples$train)


# 4. Deterministic prediction (IDW) ---------------------------------------
idw_results <- idw(target ~ 1, sample_train, sample_test)

# 5. approximate GP - train -----------------------------------------------
train <- setup_gpytorch(sample_train)
val <- setup_gpytorch(sample_val)
model_trained <- py$train_gp(
  train = train, 
  val = val, 
  epochs = 500L, 
  lr =  0.1, 
  patience = 10L, 
  save_best_model = "bestmodel.pt",
  verbose = TRUE
)

# 6. approximate GP - test ------------------------------------------------
test <- setup_gpytorch(sample_test)
yhat <- predict_gp(model = "bestmodel.pt", train = train, test = test)


# 7. Test -----------------------------------------------------------------
rmse(actual = test$target, predicted = idw_results$var1.pred)
rmse(actual = test$target, predicted = yhat$mean_values)


# 8. Prediction -----------------------------------------------------------
r_ref <- ground_truth
global_map <- setup_gpytorch(sf_coordinates)
sf_coordinates$gp_results <- py$predict_gp(model = "bestmodel.pt", train = train, test = global_map)$mean_values
sf_coordinates$idw_results <- idw(target ~ 1, sample_train, sf_coordinates)$var1.pred
stk_results <- from_sf_to_raster(sf_coordinates)
plot(stk_results[[c("gp_results", "idw_results", "target")]])


plot(stk_results[[1]])
plot(sample_train$geometry, add = TRUE, cex=0.1)

