#' Bayesian Non-Stationary Scalable Gaussian Process Model - utils
#' @author Cesar Aybar and Francois Septier

source_python("utils.py")

worldmap <- giscoR::gisco_get_countries() %>% 
  filter(NAME_ENGL!="Antarctica") %>% 
  '[['("geometry")


#' gpytorch set up!
setup_gpytorch <- function(sample, feature = "target") {
  coordinates <- st_coordinates(sample) %>% as.data.frame()
  list(features = coordinates, target = sample[[feature]])
}


#'Create a worldmap!
world_map_creator <- function(cellsize) {
  area <- st_bbox(c(xmin = 0, xmax = 360, ymax = 90, ymin = -90), crs = st_crs(4326))
  area_grid_point <- st_make_grid(area, what = "centers", cellsize)
  area_grid_point <- area_grid_point[apply(st_intersects(area_grid_point, worldmap), 1, any)]
  coordinates <- st_coordinates(area_grid_point) %>% as_tibble()
  names(coordinates) <- c("X", "Y")
  coordinates
}


#' Create a toy dataset
toy_dataset <- function(coordinates, type = "toy01") {
  N <- nrow(coordinates)
  error <- rnorm(N, 0, 0.25)
  
  b0 <- 1
  b1 <- 1 + apply(coordinates, 1, sum)/12 %>% as.numeric() # (u1+u2)/12
  b2 <- 1 + 2*cos(pi*coordinates$X/24)*cos(pi*coordinates$Y/24) # 1 + 2*cos(pi*u1/24)*cos(pi*u2/24)
  b3 <- 1 + 2*cos(pi*coordinates$X/12)*cos(pi*coordinates$Y/12) # 1 + 2*cos(pi*u1/12)*cos(pi*u2/12)
  
  if (type == "toy01") {
    coordinates$target <- as.numeric((b0 + b1 + b2 + error)) 
  }  else if (type == "toy02") {
    coordinates$target <- as.numeric(b0 + b1 + b3 + error)  
  }  else if (type == "toy03") {
    coordinates$target <- as.numeric((b0 + 4*tanh(b1/3)+ 4*tanh(b2/3))+ error)
  }  else if (type == "toy04") {
    coordinates$target <- as.numeric((b0 + 4*tanh(b1/3)+ 4*tanh(b3/3))+ error)
  } else {
    stop(sprintf("The type: %s does not exists!", type))
  }
  final_stars <- rasterFromXYZ(coordinates)
  crs(final_stars) <- 4326
  final_stars
}

block_sampling <- function(ground_truth, size = 2000, block_size = 5, test_size = 0.1, val_size = 0.1) {
  # study area box
  spbbox <- st_bbox(st_as_stars())
  # sample geographical blocks
  blocks <- spbbox %>% st_as_sfc() %>% st_make_grid(cellsize = block_size)
  blocks <- blocks[apply(st_intersects(blocks, worldmap), 1, any)]
  geo_split <- split_dataset(iterator = 1:length(blocks), test_size = test_size, val_size = val_size)
  train_sample <- geo_split[[1]]
  val_sample <- geo_split[[2]]
  test_sample <- geo_split[[3]]
  xrange <- (spbbox["xmax"] - spbbox["xmin"])*1000
  yrange <- (spbbox["ymax"] - spbbox["ymin"])*1000
  sample <- tibble(
    x = sample(xrange, size*10)/1000,
    y = sample(yrange, size*10)/1000 - 90
  ) %>% st_as_sf(coords = c("x", "y"), crs = 4326)
  sample <- extract(ground_truth, sample, sp=TRUE) %>% st_as_sf()
  sample <- na.omit(sample)
  sample <- sample[sample(nrow(sample), size),]
  sample_train <- sample[apply(st_intersects(sample, blocks[train_sample],sparse = FALSE), 1, any),]
  sample_val <- sample[apply(st_intersects(sample, blocks[val_sample],sparse = FALSE), 1, any),]
  sample_test <- sample[apply(st_intersects(sample, blocks[test_sample],sparse = FALSE), 1, any),]
  list(train = sample_train, val = sample_val, test = sample_test)  
}


from_sf_to_raster <- function(sf_coordinates) {
  sp_coordinates <- sf_coordinates %>% as("Spatial")
  gridded(sp_coordinates) <- TRUE
  stk_prediction <- stack(sp_coordinates)
  stk_prediction
}


#' FilteredTverskyMetric
#' @author Cesar Aybar
#' @examples 
#' y_true <- c(0, 0, 0 , 0, 1, 1)
#' y_pred <- c(0, 0, 0 , 1, 1, 1)
#' FilteredTverskyMetric(y_pred, y_true, alpha = 0.3, beta = 0.7)
FilteredTverskyMetric <- function(y_pred, y_true, alpha = 0.3, beta = 0.7) {
  # GLOBAL PARAMS
  KG <- 1
  KJ <- 1
  m <- 1000
  pc <- 0.5
  smooth <- 1e-5
  
  # Inverted Tversky (GL1)
  y_true_complement <- (y_true == FALSE)*1
  y_pred_complement <- 1 - y_pred
  TP <- sum(y_pred_complement * y_true_complement)
  FP <- sum((1 - y_true_complement) * y_pred_complement)
  FN <- sum(y_true_complement * (1 - y_pred_complement))
  GL <- Tversky <- (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
  
  # Standard Tversky (GL1)
  TP <- sum(y_pred * y_true)    
  FP <- sum((1 - y_true) * y_pred)
  FN <- sum(y_true * (1 - y_pred))
  JL <- Tversky <- (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
  
  FJL_01 <- (KG * GL)/(1 + exp(m*(sum(y_true) - pc)))
  FJL_02 <- (KJ * JL)/(1 + exp(m*(-sum(y_true) + pc)))
  FJL_01 + FJL_02  
}



FilteredTverskyMetric_raster <- function(y_pred, y_true, alpha = 0.3, beta = 0.7) {
  a <- y_pred[[1]] %>% as.numeric()
  b <- y_true[[1]] %>% as.numeric()
  FilteredTverskyMetric(a, b, alpha = alpha, beta = beta)
}
