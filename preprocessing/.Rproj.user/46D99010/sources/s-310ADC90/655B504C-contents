#' Bayesian Non-Stationary Scalable Gaussian Process Model - utils
#' @author Cesar Aybar and Francois Septier

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

FilteredTverskyMetric2 <- function(y_pred, y_true, alpha = 0.3, beta = 0.7) {
  if (any(y_pred == 1) & any(y_true == 1)) {
    FilteredTverskyMetric(y_pred, y_true, alpha = alpha, beta = beta)
  } else {
    NA
  }
}

generate_dataset_predictors <- function(output = "cloud_dt.csv") {
  container <- list()
  for (index in 1:2000) {
    print(index)
    ROI_NAME <- gsub(pattern = "__(ROI_.*)__.*", "\\1", basename(GT_FILES$hq[(index - 1)*5 + 1]))  
    S2_NAME <- gsub(pattern = "__ROI_.*__(.*)\\.tif$", "\\1", basename(GT_FILES$hq[(index - 1)*5 + 1]))
    FOLDER_TYPE <- GT_FILES$type[(index - 1)*5 + 1]
    ref_file <- GT_FILES$hq[grep(ROI_NAME, GT_FILES$hq)[1]]
    reference <- read_stars(ref_file) %>%
      st_bbox() %>%
      st_as_sfc() %>%
      st_transform(4326)
    
    # Direct normal irradiation - Longterm average of direct normal irradiation
    # For additional information: https://globalsolaratlas.info
    var_dni <- ee$Image('projects/earthengine-legacy/assets/projects/sat-io/open-datasets/global_solar_atlas/dni_LTAy_AvgDailyTotals');
    var_dni_value <- ee$Image$reduceRegion(
      image = var_dni,
      reducer = ee$Reducer$mean(),
      geometry = sf_as_ee(reference)
    )$getInfo()[[1]]
    
    # cloud cover 1km - mean
    # Mean annual cloud frequency (%) over 2000-2014, . Valid values range from 0-
    # 10,000 and need to be multiplied by 0.01 to result in % cloudy days. Values greater than
    # 10,000 are used for fill.
    var_mean_cloud_annual <- ee$Image("projects/sat-io/open-datasets/gcc/MODCF_meanannual")
    var_mean_cloud_annual_value <- ee$Image$reduceRegion(
      image = var_mean_cloud_annual,
      reducer = ee$Reducer$mean(),
      geometry = sf_as_ee(reference)
    )$getInfo()[[1]]*0.01
    
    # MODCF_intraannualSD
    # Within-year seasonality represented as the standard deviation of mean 2000-2014 monthly
    # cloud frequencies, SD(μm). Values need to be multiplied by 0.01 to recover SD
    var_intra_cloud_annual <- ee$Image("projects/sat-io/open-datasets/gcc/MODCF_intraannualSD")
    var_intra_cloud_annual_value <- ee$Image$reduceRegion(
      image = var_intra_cloud_annual,
      reducer = ee$Reducer$mean(),
      geometry = sf_as_ee(reference)
    )$getInfo()[[1]]*0.01
    
    # DEM - MERIT
    var_dem <- ee$Image("MERIT/Hydro/v1_0_1")$select("elv")
    var_dem_value <- ee$Image$reduceRegion(
      image = var_dem,
      reducer = ee$Reducer$mean(),
      geometry = sf_as_ee(reference)
    )$getInfo()[[1]]
    
    # Land USE 
    FOLDER_HQ <- "/media/csaybar/Elements SE/cloudSEN12/"
    luse_file <- read_stars(sprintf("%s/%s/%s/%s/extra/LC10.tif", FOLDER_HQ, FOLDER_TYPE,ROI_NAME, S2_NAME))[[1]]
    
    classes <- list(
      "forest" = 10,
      "Barren / sparse vegetation" = c(20, 30, 60, 90, 95, 100),
      "cropland" = 40,
      "Built-up" = 50,
      "Snow and ice" = 70,
      "Open water" = 80
    )
    
    if (!any(unique(luse_file[[1]] %>% as.numeric()) %in%  as.numeric(unlist(classes)))) {
      stop("Check ROI :c")
    }
    
    luse_file[luse_file == 10] <- 0
    luse_file[luse_file  %in%  c(20, 30, 60, 90, 95, 100)] <- 1
    luse_file[luse_file == 40] <- 2
    luse_file[luse_file == 50] <- 3
    luse_file[luse_file == 70] <- 4
    luse_file[luse_file == 80] <- 5
    
    # values
    total_prob <- table(luse_file)/(509*509)
    
    # SNOW PROB
    var_snow_lt_p90 <- ee$ImageCollection("projects/sat-io/open-datasets/MODIS_LT_SNOW/monthly_lt_p90")$mean()
    var_snow_lt_p90_value <- ee$Image$reduceRegion(
      image = var_snow_lt_p90,
      reducer = ee$Reducer$mean(),
      geometry = sf_as_ee(reference),
      scale = 100
    )$getInfo()[[1]]
    
    
    container[[index]] <- list(
      x = st_centroid(reference)[[1]][1],
      y = st_centroid(reference)[[1]][2],
      ROI_ID = ROI_NAME,
      direct_normal_irradiation = var_dni_value,
      mean_cloud_annual = var_mean_cloud_annual_value,
      intra_cloud_annual = var_intra_cloud_annual_value,
      elevation = var_dem_value,
      forest_prob = if (is.na(total_prob["1"])) 0 else total_prob["1"],
      sveg_prob = if (is.na(total_prob["2"])) 0 else total_prob["2"],
      cropland_prob = if (is.na(total_prob["3"])) 0 else total_prob["3"],
      builup_prob = if (is.na(total_prob["4"])) 0 else total_prob["4"],
      snow_prob = if (is.na(total_prob["5"])) 0 else total_prob["5"],
      water_prob = if (is.na(total_prob["6"])) 0 else total_prob["6"],
      var_snow_lt_p90 = var_snow_lt_p90_value
    )
  }
  
  db_merged <- do.call(bind_rows, container)
  write_csv(
    x = db_merged,
    file = output
  )
  st_as_sf(db_merged, coords = c("x", "y"), crs = 4326)
}


generate_dataset_target <- function(cd_model = "sen2cor", output = "cloud_target.csv") {
  S2COR_FILES <- sprintf("%s/%s/hq/%s", CLOUD_MODEL_FOLDER, cd_model, CLOUDSEN12_IDS)
  unique_rois <- unique(gsub("__(ROI_.*)__.*\\.tif", "\\1", basename(S2COR_FILES)))
  
  # cot - sen2cor - TARGET
  sen2cor_cot <- foreach(index=1:2000, .combine = bind_rows) %dopar% {
    roi_id <- unique_rois[index]
    gt_val <- names(get_groups(roi_id))
    sen2cor_val <- gsub("z_target", "sen2cor",gt_val)
    # grounth truth as star object
    r1 <- transform_gt(gt_val)
    
    # sen2cor as star object
    r2 <- transform_sen2cor(sen2cor_val)
    # plot(r1[[2]])
    # plot(r2[[2]])
    
    # TOTAL PIXELS
    r1_val <- sapply(r1, function(x) x[[1]]) %>% as.numeric()
    r2_val <- sapply(r2, function(x) x[[1]]) %>% as.numeric()
    total_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    total_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    total_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # cloud-free PIXELS
    r1_val <- r1[[1]][[1]] %>% as.numeric()
    r2_val <- r2[[1]][[1]] %>% as.numeric()
    CF_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    CF_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    CF_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # almost-clear PIXELS
    r1_val <- r1[[2]][[1]] %>% as.numeric()
    r2_val <- r2[[2]][[1]] %>% as.numeric()
    AC_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    AC_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    AC_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # low-cloudy PIXELS
    r1_val <- r1[[3]][[1]] %>% as.numeric()
    r2_val <- r2[[3]][[1]] %>% as.numeric()
    LC_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    LC_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    LC_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # mid-cloudy PIXELS
    r1_val <- r1[[4]][[1]] %>% as.numeric()
    r2_val <- r2[[4]][[1]] %>% as.numeric()
    MC_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    MC_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    MC_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # cloudy PIXELS
    r1_val <- r1[[5]][[1]] %>% as.numeric()
    r2_val <- r2[[5]][[1]] %>% as.numeric()
    C_clear <- FilteredTverskyMetric2(r1_val == 0, r2_val == 0)
    C_cloud <- FilteredTverskyMetric2(r1_val == 1, r2_val == 1)
    C_cs <- FilteredTverskyMetric2(r1_val == 2, r2_val == 2)
    
    # Cloud Optimized GEOTIFF
    list(
      ROI_ID = roi_id,
      total_clear = total_clear,
      total_cloud = total_cloud,
      total_cs = total_cs,
      CF_clear = CF_clear,
      CF_cloud = CF_cloud,
      CF_cs = CF_cs,
      AC_clear = AC_clear,
      AC_cloud = AC_cloud,
      AC_cs = AC_cs,
      LC_clear = LC_clear,
      LC_cloud = LC_cloud,
      LC_cs = LC_cs,
      MC_clear = MC_clear,
      MC_cloud = MC_cloud,
      MC_cs = MC_cs,
      C_clear = C_clear,
      C_cloud = C_cloud,
      C_cs = C_cs
    )
  }
  
  write_csv(
    x = sen2cor_cot,
    file = output
  )
  sen2cor_cot
}


get_groups <- function(roi_id) {
  get_probs <- function(x) {
    x_r <- read_stars(x)
    result <- sum((x_r == 1)[[1]])/(509*509)
    result
  }
  sort(sapply(GT_FILES$hq[grep(roi_id, GT_FILES$hq)], get_probs))
}


transform_gt <- function(gt_val) {
  transform <- function(x) {
    gt <- read_stars(x)
    gt_clear <- (gt == 0)*1
    gt_cloud <- (gt == 1 | gt == 2)*2
    gt_cloud_shadow <- (gt == 3)*3
    (gt_clear + gt_cloud + gt_cloud_shadow) - 1
  }
  stars_obj <- lapply(gt_val, transform)
}

transform_sen2cor <- function(gt_val) {
  transform <- function(x) {
    cloud_detector <- read_stars(x)
    pd_clear <- (cloud_detector == 4 | cloud_detector == 5 | cloud_detector == 6 | cloud_detector == 11 | cloud_detector == 2)*1
    pd_cloud <- (cloud_detector == 8 | cloud_detector == 9 | cloud_detector == 10 | cloud_detector == 7)*2
    pd_cloud_shadow <- (cloud_detector == 3)*3
    result <- pd_clear + pd_cloud + pd_cloud_shadow - 1
    result[result < 0 ] = NA
    result
  }
  stars_obj <- lapply(gt_val, transform)
}

