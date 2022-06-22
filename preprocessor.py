import pandas as pd

class PreProcessor:
    def __init__(self, dataset, model_type, target_name="target") -> None:
        self.dataset = dataset
        self.target_name = target_name
        self.model_type = model_type
        self.cat_cols = None
        self.num_cols = None

    def pre_processor(self):
        if self.model_type == "regression":
            # do this
            pass
        else:
            # do this
            pass

        dataset = self.process_imputation(self.dataset)
        dataset = self.process_skew(dataset)
        datatet = self.process_encoding(dataset[self.cat_cols])
        dataset = self.process_scaling(dataset[self.num_cols])


    def process_imputation(self, dataset):
        pass
    
    def process_skew(self):
        pass

    def process_encoding(self):
        pass

    def process_scaling(self):
        pass