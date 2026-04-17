import pandas as pd
from sklearn.base import BaseEstimator
import joblib as job

class Shelf:
    def __init__(self,  
                 X_train : pd.DataFrame,
                 X_test : pd.DataFrame,
                 model : BaseEstimator,
                 path : str,
                 **kwargs):
        """
        Docstring for __init__
        
        :param self: This Shelf object
        :param model: your model
        :type model: BaseEstimator
        :param X_train: self explanatory
        :type X_train: pd.DataFrame
        :param X_test: self explanatory
        :type X_test: pd.DataFrame
        :param path: Where should I save the pkl
        :type path: str
        :param kwargs: Add here every extra thing you want to save into the pkl. You will need to load with return_object = True to retrieve it.
        """
        
        self.train_index = X_train.index
        self.test_index = X_test.index
        self.features = X_train.columns
        self.model = model
        self.path = path

        self.extra = kwargs

        job.dump(self, self.path)
    
    @classmethod
    def load(cls, X : pd.DataFrame, y : pd.DataFrame, path : str, ReturnMultiIndex : bool = True, return_extra : bool = False):
        """
        Docstring for load
        
        :param X: Your X
        :type X: pd.DataFrame
        :param y: Your y
        :type y: pd.DataFrame
        :param path: Where should I load the pkl from?
        :type path: str
        :param ReturnMultiIndex: Should I return the y_train and y_test as multiIndex? [True]
        :type ReturnMultiIndex: bool
        :param return_object: Should I return the object instead? Especially useful when you save stuff under extra
        :type return_object: bool
        :param returns:  X_train, X_test, y_train, y_test, model
        """


        obj : Shelf = job.load(path)
        obj.train_index = obj.train_index.set_levels(
            [obj.train_index.levels[0].astype(int),
            obj.train_index.levels[1]], 
            level=[0, 1])
        obj.test_index = obj.test_index.set_levels(
            [obj.test_index.levels[0].astype(int),
            obj.test_index.levels[1]],
            level=[0, 1])
    
        print(f"Test set videos are:    {obj.test_index.get_level_values(0).unique().to_numpy()}")

        X_train = X.loc[obj.train_index][obj.features]
        X_test = X.loc[obj.test_index][obj.features]
        y_train = y.loc[obj.train_index]
        y_test = y.loc[obj.test_index]

        if not ReturnMultiIndex:
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
        
        if return_extra:
            return X_train, X_test, y_train, y_test, obj.model, obj.extra
        
        return X_train, X_test, y_train, y_test, obj.model

