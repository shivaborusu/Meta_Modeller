import pandas as pd
def pre_processor(dataset, target="target"):
    # implement seperation of int/float and object columns
    numerical_cols = dataset.select_dtypes(exclude=object).columns
    cat_cols = dataset.select_dtypes(include=object).columns