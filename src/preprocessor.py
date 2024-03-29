import pandas as pd
import numpy as np
from config import SEED, DATA_SET_PATH, MODEL_PICKLE_PATH
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer, LabelEncoder
import category_encoders as ce


class PreProcessor:
    """
    The PreProcessor class follows the guidance of CRISP DM 
    to clean the data suitable to be consumed by any Machine Learning Model

    This class currently deals with both Numerical and Categorical 
    data cleaning.

    The method mentioned here are a sequecnce of steps which takes
    an intitial data frame and outputs it as a feature space with obeservations
    
    """
    def __init__(self, dataset, model_type, target_name="target") -> None:
        """
        Class constructor

        Inputs:
        dataset: raw dataset input
        model_type: species the model is a regressor or classifier
        target_name: name of the target in feature space, defaults to "target"

        """
        self.dataset = dataset
        self.target_name = target_name
        self.model_type = model_type
        self.cat_cols = None
        self.num_cols = None

    def pre_processor(self):  
        """
        The orchestrator method which ties together all the other member
        methods inside the class. This method demonstrates a pipeline to handle all the
        preprocessing tasks required for a machine learning model building
        """
        dataset = self.dataset.copy()

        dataset = self.process_trivial_cleanup(dataset)

        # These intermediate saving is needed because 
        # the initial dataset has '?' symbol in numerical column
        # so it is read as 'object' column
        # After replacing '?' with np.NaN it won't change the datatype
        # so saving the dataframe to csv and reloading it will make the column 
        # numrical type
        # this saving and reloading converting numerical type columns to string type
        dataset.to_csv(DATA_SET_PATH + "regression/car_prices/car.csv", index=False)
        dataset = pd.read_csv(DATA_SET_PATH + "regression/car_prices/car.csv")

        # seperate out target data
        target = dataset[[str(self.target_name)]]
        dataset = dataset.drop(str(self.target_name), axis=1)

        dataset_cat, dataset_num = self.process_cat_num(dataset)

        #dataset = self.process_skew(dataset)
        #dataset_cat = self.process_encoding(dataset[self.cat_cols])
        #dataset_num = self.process_scaling(dataset[self.num_cols])

        if not dataset_cat.empty:
            x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test =\
                train_test_split(dataset_cat, dataset_num, target, random_state=SEED)

            x_train_cat, x_test_cat, x_train_num, x_test_num =\
                self.process_imputation(x_train_cat, x_test_cat, x_train_num, x_test_num)

            x_train_num, x_test_num = self.process_num(x_train_num, x_test_num)
            x_train_cat, x_test_cat = self.process_cat(x_train_cat, x_test_cat, y_train)

            # these lines are inserted here to test KNN imputation strategy on encoded categorical variables
            # x_train_cat, x_test_cat, x_train_num, x_test_num =\
            #    self.process_imputation_2(x_train_cat, x_test_cat, x_train_num, x_test_num)

            x_train_df, x_test_df =\
                self.process_merge(x_train_num, x_test_num, x_train_cat, x_test_cat)

            # assigning original column names
            cols_list = dataset_num.columns.tolist() + dataset_cat.columns.tolist()
            x_train_df.columns = cols_list
            x_test_df.columns = cols_list

        else:
            print("No Category Columns, only numerical processing will run...")
            x_train, x_test, y_train, y_test =\
                train_test_split(dataset_num, target, random_state=SEED)

            x_train, x_test =\
                self.process_imputation_num(x_train, x_test)

            x_train, x_test= self.process_num(x_train, x_test)

            x_train_df, x_test_df =\
                self.process_merge(x_train, x_test)

            cols_list = dataset_num.columns.tolist()
            x_train_df.columns = cols_list
            x_test_df.columns = cols_list
        
        # saving x_test_df to verify it in flask UI
        x_train_df.to_csv(MODEL_PICKLE_PATH + "Test_Files/" + "pp_train_df.csv", index=False, header=True)
        x_test_df.to_csv(MODEL_PICKLE_PATH + "Test_Files/" + "pp_test_df.csv", index=False, header=True)

        return x_train_df, x_test_df, y_train, y_test, cols_list


    def process_trivial_cleanup(self, dataset):
        """
        This method handles dataset specific cleaning. This
        method can be further modified in order to handle trivial issues corresponding to
        different datasets. For example, in one of the datasets the missing values are
        represented with a '?', this method replaces this '?' with a 'np.NaN'. This helps to
        get the type of the column correctly which further helps in imputation/handling
        missing values.
        """
        # rename target for consistency
        #dataset = dataset.rename(columns={self.target_name:"target"})
        dataset = dataset.replace("?", np.NaN)
        dataset = dataset.replace(np.inf, np.NaN)
        dataset = dataset[dataset[self.target_name].isnull() == False].reset_index(drop=True)

        return dataset


    def process_cat_num(self, dataset):
        """
        Responsible for separating numerical and categorical columns.
        """
        cat_cols = dataset.select_dtypes(include='object').columns.tolist()
        num_cols = dataset.select_dtypes(exclude='object').columns.tolist()
        
        # if there are no categorical variables
        if not cat_cols:
            return pd.DataFrame(), dataset[num_cols]
        else:
            dataset_cat = dataset[cat_cols]
            dataset_num = dataset[num_cols]

        return dataset_cat, dataset_num


    def process_imputation_num(self, x_train_num, x_test_num):
        """
        Responsible for handling missing values. This can handle
        both numerical and categorical columns
        """

        imputer_num = KNNImputer()

        x_train_num = imputer_num.fit_transform(x_train_num)
        x_test_num = imputer_num.transform(x_test_num)

        return x_train_num, x_test_num


    def process_imputation(self, x_train_cat, x_test_cat, x_train_num, x_test_num):
        """
        Responsible for handling missing values. This can handle
        both numerical and categorical columns
        """

        imputer_cat = SimpleImputer(strategy='constant', fill_value='NA')
        imputer_num = KNNImputer()

        x_train_cat = imputer_cat.fit_transform(x_train_cat)
        x_test_cat = imputer_cat.transform(x_test_cat)

        x_train_num = imputer_num.fit_transform(x_train_num)
        x_test_num = imputer_num.transform(x_test_num)

        return x_train_cat, x_test_cat, x_train_num, x_test_num    
        
    
    def process_num(self, x_train_num, x_test_num):
        """
        Responsible for handling skew and normalisation of numerical
        features. Which orchestrates process_skew and process_scaling methods
        """
        # Inside process skew we are already doing scaling by using
        # Standardize = True, we can skip process scaling
        # process_scaling is just kept for descriptive purposes
        x_train_num, x_test_num = self.process_skew(x_train_num, x_test_num)
        x_train_num, x_test_num = self.process_scaling(x_train_num, x_test_num)

        return x_train_num, x_test_num

    def process_cat(self, x_train_cat, x_test_cat, y_train):
        """
        Responsible for handling categorical features. Orchestrates
        process_encoding to encode categorical features. This can be extended to support
        further categorical features processing
        """
        x_train_cat, x_test_cat = self.process_encoding(x_train_cat, x_test_cat, y_train)

        return x_train_cat, x_test_cat

    def process_skew(self, x_train_num, x_test_num):
        """
        Handles the data skew related to numerical features
        """
        yeo_john_pt = PowerTransformer(method='yeo-johnson', standardize=True)
        x_train_num = yeo_john_pt.fit_transform(x_train_num)
        x_test_num = yeo_john_pt.transform(x_test_num)

        return x_train_num, x_test_num

    def process_scaling(self, x_train_num, x_test_num):
        """
        Scaling is handled in process_skew by setting the parameter
        standardize=True
        """
        return x_train_num, x_test_num

    def process_encoding(self, x_train_cat, x_test_cat, y_train):
        tenc = ce.TargetEncoder(min_samples_leaf=20, smoothing=10)

        x_train_cat = tenc.fit_transform(x_train_cat, y_train.reset_index(drop=True))
        x_test_cat = tenc.transform(x_test_cat)

        return x_train_cat, x_test_cat

    def process_merge(self, x_train_num, x_test_num, x_train_cat=pd.DataFrame(), x_test_cat=pd.DataFrame()):
        """
        Responsible for stitching the processed numerical and categorical
        features together
        """

        if not x_train_cat.empty:
            # combining categorical and numerical features in train, test
            x_train = np.hstack([x_train_cat, x_train_num])
            x_test = np.hstack([x_test_cat, x_test_num])
        else:
            x_train = x_train_num
            x_test = x_test_num

        # writing to CSV and reloading from CSV is performed 
        # because numpy columns are read as all 'object' type
        x_train_df = pd.DataFrame(x_train, dtype=None)
        x_train_df.to_csv(DATA_SET_PATH + "regression/car_prices/x_train_df.csv", index=False)
        x_train_df = pd.read_csv(DATA_SET_PATH + "regression/car_prices/x_train_df.csv")


        x_test_df = pd.DataFrame(x_test, dtype=None)
        x_test_df.to_csv(DATA_SET_PATH + "regression/car_prices/x_test_df.csv", index=False)
        x_test_df = pd.read_csv(DATA_SET_PATH + "regression/car_prices/x_test_df.csv")


        # this step is needed because 'object' is not recorgnized by LGBM
        if not x_train_cat.empty:
            cat_cols = x_train_df.select_dtypes(include='object').columns
            for col in cat_cols:
                x_train_df[col] = x_train_df[col].astype('category')
                x_test_df[col] = x_test_df[col].astype('category')

        return x_train_df, x_test_df
