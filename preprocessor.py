import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from config import SEED


class PreProcessor:
    def __init__(self, dataset, model_type, target_name="target") -> None:
        self.dataset = dataset
        self.target_name = target_name
        self.model_type = model_type
        self.cat_cols = None
        self.num_cols = None
        self.seed = np.random.randint(1,100)

    def pre_processor(self):
        dataset = self.dataset.copy()

        # rename target for consistency
        #dataset = dataset.rename(columns={self.target_name:"target"})
        dataset = dataset[dataset[self.target_name].isnull() == False]
        dataset = dataset.replace("?", np.NaN)

        # These intermediate saving is needed because 
        # the initial dataset has '?' symbol in numerical column
        # so it is read as 'object' column
        # After replacing '?' with np.NaN it won't change the datatype
        # so saving the dataframe to csv and reloading it will make the column 
        # numrical type
        # this saving and reloading converting numerical type columns to string type
        dataset.to_csv("./datasets/regression/car_prices/car.csv", index=False)
        dataset = pd.read_csv("./datasets/regression/car_prices/car.csv")

        # seperate out target data

        target = dataset[[str(self.target_name)]]
        dataset = dataset.drop(str(self.target_name), axis=1)

        dataset_cat, dataset_num = self.process_cat_num(dataset)

        #dataset = self.process_skew(dataset)
        #dataset_cat = self.process_encoding(dataset[self.cat_cols])
        #dataset_num = self.process_scaling(dataset[self.num_cols])

        x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test =\
            train_test_split(dataset_cat, dataset_num, target, random_state=SEED)

        x_train_df, x_test_df = self.process_imputation(x_train_cat, x_test_cat, x_train_num, x_test_num)    

        return x_train_df, x_test_df, y_train, y_test


    def process_cat_num(self, dataset):
        cat_cols = dataset.select_dtypes(include='object').columns
        num_cols = dataset.select_dtypes(exclude='object').columns
        dataset_cat = dataset[cat_cols]
        dataset_num = dataset[num_cols]

        return dataset_cat, dataset_num


    def process_imputation(self, x_train_cat, x_test_cat, x_train_num, x_test_num):
        imputer_cat = SimpleImputer(strategy='constant', fill_value='NA')
        imputer_num = SimpleImputer(strategy='median')

        x_train_cat = imputer_cat.fit_transform(x_train_cat)
        x_test_cat = imputer_cat.transform(x_test_cat)

        x_train_num = imputer_num.fit_transform(x_train_num)
        x_test_num = imputer_num.transform(x_test_num)

        # combining categorical and numerical features in train, test
        x_train = np.hstack([x_train_cat, x_train_num])
        x_test = np.hstack([x_test_cat, x_test_num])

        # writing to CSV and reloading from CSV is performed 
        # because numpy columns are read as all 'object' type
        x_train_df = pd.DataFrame(x_train, dtype=None)
        x_train_df.to_csv("./datasets/regression/car_prices/x_train_df.csv", index=False)
        x_train_df = pd.read_csv("./datasets/regression/car_prices/x_train_df.csv")


        x_test_df = pd.DataFrame(x_test, dtype=None)
        x_test_df.to_csv("./datasets/regression/car_prices/x_test_df.csv", index=False)
        x_test_df = pd.read_csv("./datasets/regression/car_prices/x_test_df.csv")

        # this step is needed because 'object' is not recorgnized by LGBM
        cat_cols = x_train_df.select_dtypes(include='object').columns
        for col in cat_cols:
            x_train_df[col] = x_train_df[col].astype('category')
            x_test_df[col] = x_test_df[col].astype('category')

        return x_train_df, x_test_df
    
    def process_skew(self):
        pass

    def process_encoding(self):
        pass

    def process_scaling(self):
        pass