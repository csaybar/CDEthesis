from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pprint import pprint
from scipy import stats
import seaborn as sns
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import datetime
import requests
import re


dataset = pd.read_csv("cloud_db.csv")

X = dataset[[
    "mean_cloud_annual", "intra_cloud_annual", "elevation", 
    "forest_prob", "sveg_prob", "cropland_prob", 
    "builup_prob", "snow_prob", "water_prob", "var_snow_lt_p90",
    "direct_normal_irradiation"
]]

y = dataset["AC_clear"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
train_test_RF_local(X_train, X_test, y_train, y_test)
