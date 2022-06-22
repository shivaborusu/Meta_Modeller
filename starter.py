# imports
import pandas as pd
import numpy as np
from preprocessor import pre_processor


# read the data set

dataset = pd.read_csv("Meta_Modeller/datasets/regression/car_prices/imports-85.data", header=None)


train, test, validation = pre_processor(dataset, target_name=None)