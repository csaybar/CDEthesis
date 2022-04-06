#' @title How reliable are Sentinel-2 cloud detection algorithms?: Global 
#' uncertainty estimation with GP.
#' @description Generating tabular dataset for csaybar thesis
#' @author Cesar Aybar
#' 

library(sf)
library(rgee)
library(doMC)
library(dplyr)
library(stars)
library(stars)
library(foreach)
library(tidyverse)
require(MLmetrics)
library(rgeeExtra)
library(reticulate)

ee_Initialize()

registerDoMC(12)

source("utils.R")

# 1. global parameters -------------------------------------------------------
CLOUD_MODEL_FOLDER <- "/home/csaybar/Documents/Github/thesis/preprocessing/models/"
CLOUDSEN12_METADATA <- read_csv("/media/csaybar/58059B472A3AA231/cloudsen12_metadata_v8.csv")
CLOUDSEN12_METADATA <- CLOUDSEN12_METADATA[CLOUDSEN12_METADATA$label_type == "high",]
CLOUDSEN12_IDS <- sprintf("__%s__%s.tif", CLOUDSEN12_METADATA$roi_id, CLOUDSEN12_METADATA$s2_id_gee)
GT_FILES <- list(
  hq = sprintf("%s/z_target/hq/%s", CLOUD_MODEL_FOLDER, CLOUDSEN12_IDS),
  sc = sprintf("%s/z_target/scribble/%s", CLOUD_MODEL_FOLDER, CLOUDSEN12_IDS),
  type = CLOUDSEN12_METADATA$label_type
)


# 3. Create grid prediction -----------------------------------------------
world_grid <- world_map_creator(cellsize = 0.25)


# 2. Generate predictors ---------------------------------------------------
dataset_predictors <- generate_dataset_predictors(output = "cloud_dt.csv")
# dataset_predictors <- st_as_sf(read_csv("cloud_dt.csv"), coords = c("x", "y"), crs = 4326)

# 3. Add target --------------------------------------------------------------
dataset_target <- generate_dataset_target(cd_model = "sen2cor", output = "cloud_target.csv")
full_dataset <- merge(dataset_predictors, dataset_target, by = "ROI_ID")

write_csv(
  x = full_dataset,
  file = "cloud_db.csv"
)


mapview::mapview(full_dataset[c("AC_cloud", "ROI_ID")], zcol = "AC_cloud", legend = FALSE)



ggp <-CLOUDSEN12_METADATA[CLOUDSEN12_METADATA$roi_id == "ROI_7769" & CLOUDSEN12_METADATA$cloud_coverage == "almost-clear",]
r_files <- sprintf("/media/csaybar/Elements SE/cloudSEN12/high/%s/%s/labels/%s", ggp$roi_id, ggp$s2_id_gee, c("sen2cor.tif", "manual_hq.tif"))
RGB_ref <- sprintf("%s/S2L1C.tif", dirname(dirname(r_files))[1])
sen2cor_r <- transform_sen2cor(r_files[1])[[1]] %>% as("Raster")
sen2cor_r[sen2cor_r == 0] = NA
gt_r <- transform_gt(r_files[2])[[1]] %>% as("Raster")
gt_r[gt_r == 0] = NA


plotRGB(stack(RGB_ref)[[c("B4", "B3", "B2")]], stretch = "lin")
plot(sen2cor_r, add = TRUE)
plot(gt_r, add = TRUE)
