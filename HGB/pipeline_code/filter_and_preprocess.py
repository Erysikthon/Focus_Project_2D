import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from imblearn.under_sampling import RandomUnderSampler

def reduce_bits(X : pd.DataFrame):
    float64_cols = list(X.select_dtypes(include='float64'))
    X[float64_cols] = X[float64_cols].astype('float32')
    return X

def collinearity_filter(X: pd.DataFrame, threshold=0.95):
    """
    Pure collinearity filtering based only on feature correlations.
    Does not use target variable, so safe to apply before train/test split.

    For each group of correlated features, keeps the first one and drops the rest.
    """
    correlation_matrix = X.corr().abs()

    # Get upper triangle of correlation matrix
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    print(f"Dropping {len(to_drop)} highly correlated features (threshold={threshold})")

    return X.drop(columns=to_drop)

def scale(X_train, X_test):
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train, X_test

class SmartCollinearityFilter:
    def __init__(self, threshold = 0.95, uvfs_method = f_classif):

        self.threshold = threshold
        self.uvfs_method = uvfs_method

    def fit(self,
            X_train : pd.DataFrame,
            y_train : pd.DataFrame):
        
        X_train = pd.DataFrame(X_train)
        correlation_matrix = X_train.corr()
        relevant_matrix = []
        selector = SelectKBest(score_func=f_classif, k = "all")
        selector.fit(X_train, y_train)
        scores = selector.scores_
        norm_scores = scores / np.max(scores)
        feature_importance_dataframe = pd.DataFrame({"feature":pd.Series(X_train.columns),"score":pd.Series(norm_scores)})

        for row in correlation_matrix:
            temp = correlation_matrix[row].sort_values(axis=0,ascending=False)
            temp.pop(temp.name)
            temp = temp[temp > self.threshold]

            if temp.size>0:
                already_contained=False
                series = []
                series.append(temp.name)
                for item in temp.index:
                    series.append(item)
                series.sort()
                for element in series:
                    for series2 in relevant_matrix:
                        for element2 in series2:
                            if element2==element:
                                if len(series2)>=len(series):
                                    already_contained = True
                                else:
                                    relevant_matrix.remove(series2)
                if not already_contained:
                    relevant_matrix.append(series)
        self.relevant_matrix = relevant_matrix
        self.feature_importance_dataframe = feature_importance_dataframe
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        all_cols_to_drop = []
        for series in self.relevant_matrix:
            todrop = self.feature_importance_dataframe[self.feature_importance_dataframe["feature"].isin(series)]
            todrop = todrop.sort_values(by="score", ascending=False)
            all_cols_to_drop.extend(todrop["feature"].iloc[1:].tolist())
        X = X.drop(columns=list(set(all_cols_to_drop)), errors="ignore")
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame):
        self.fit(X, y)
        return self.transform(X)