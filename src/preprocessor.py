import pandas as pd
import numpy as np
from config import SEED
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
import category_encoders as ce


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

        dataset = self.process_trivial_cleanup(dataset)

        # These intermediate saving is needed because 
        # the initial dataset has '?' symbol in numerical column
        # so it is read as 'object' column
        # After replacing '?' with np.NaN it won't change the datatype
        # so saving the dataframe to csv and reloading it will make the column 
        # numrical type
        # this saving and reloading converting numerical type columns to string type
        dataset.to_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/car.csv", index=False)
        dataset = pd.read_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/car.csv")

        # seperate out target data
        target = dataset[[str(self.target_name)]]
        dataset = dataset.drop(str(self.target_name), axis=1)

        dataset_cat, dataset_num = self.process_cat_num(dataset)

        #dataset = self.process_skew(dataset)
        #dataset_cat = self.process_encoding(dataset[self.cat_cols])
        #dataset_num = self.process_scaling(dataset[self.num_cols])

        x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test =\
            train_test_split(dataset_cat, dataset_num, target, random_state=SEED)

        x_train_cat, x_test_cat, x_train_num, x_test_num =\
            self.process_imputation(x_train_cat, x_test_cat, x_train_num, x_test_num)

        x_train_num, x_test_num = self.process_num(x_train_num, x_test_num)
        x_train_cat, x_test_cat = self.process_cat(x_train_cat, x_test_cat, y_train)

        x_train_df, x_test_df =\
            self.process_merge(x_train_num, x_test_num, x_train_cat, x_test_cat)

        return x_train_df, x_test_df, y_train, y_test


    def process_trivial_cleanup(self, dataset):
        # rename target for consistency
        #dataset = dataset.rename(columns={self.target_name:"target"})
        dataset = dataset.replace("?", np.NaN)
        dataset = dataset.replace(np.inf, np.NaN)
        dataset = dataset[dataset[self.target_name].isnull() == False]

        return dataset


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

        return x_train_cat, x_test_cat, x_train_num, x_test_num
        
    
    def process_num(self, x_train_num, x_test_num):
        # Inside process skew we are already doing scaling by using
        # Standardize = True, we can skip process scaling
        # process_scaling is just kept for descriptive purposes
        x_train_num, x_test_num = self.process_skew(x_train_num, x_test_num)
        x_train_num, x_test_num = self.process_scaling(x_train_num, x_test_num)

        return x_train_num, x_test_num

    def process_cat(self, x_train_cat, x_test_cat, y_train):
        x_train_cat, x_test_cat = self.process_encoding(x_train_cat, x_test_cat, y_train)

        return x_train_cat, x_test_cat

    def process_skew(self, x_train_num, x_test_num):
        yeo_john_pt = PowerTransformer(method='yeo-johnson', standardize=True)
        x_train_num = yeo_john_pt.fit_transform(x_train_num)
        x_test_num = yeo_john_pt.transform(x_test_num)

        return x_train_num, x_test_num

    def process_scaling(self, x_train_num, x_test_num):
        return x_train_num, x_test_num

    def process_encoding(self, x_train_cat, x_test_cat, y_train):
        tenc = ce.TargetEncoder(min_samples_leaf=20, smoothing=10)

        x_train_cat = tenc.fit_transform(x_train_cat, y_train.reset_index(drop=True))
        x_test_cat = tenc.transform(x_test_cat)

        return x_train_cat, x_test_cat

    def process_merge(self, x_train_num, x_test_num, x_train_cat, x_test_cat):
        # combining categorical and numerical features in train, test
        x_train = np.hstack([x_train_cat, x_train_num])
        x_test = np.hstack([x_test_cat, x_test_num])

        # writing to CSV and reloading from CSV is performed 
        # because numpy columns are read as all 'object' type
        x_train_df = pd.DataFrame(x_train, dtype=None)
        x_train_df.to_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/x_train_df.csv", index=False)
        x_train_df = pd.read_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/x_train_df.csv")


        x_test_df = pd.DataFrame(x_test, dtype=None)
        x_test_df.to_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/x_test_df.csv", index=False)
        x_test_df = pd.read_csv("/Users/shivaborusu/Development/Meta_Modeller/datasets/regression/car_prices/x_test_df.csv")

        # this step is needed because 'object' is not recorgnized by LGBM
        cat_cols = x_train_df.select_dtypes(include='object').columns
        for col in cat_cols:
            x_train_df[col] = x_train_df[col].astype('category')
            x_test_df[col] = x_test_df[col].astype('category')

        return x_train_df, x_test_df