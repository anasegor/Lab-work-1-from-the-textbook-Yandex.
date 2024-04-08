from scipy.stats import mode
import numpy as np
from sklearn.base import RegressorMixin


class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        self.param = y.mean()
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self.param)


from sklearn.base import ClassifierMixin


class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        self.most_frequent_class_ = np.argmax(np.bincount(y))
        return self

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self.most_frequent_class_)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        d1 = {"msk": False, "spb": True}
        X1 = X["city"].map(d1)
        msk_count = X1[X1 == False].count()
        spb_count = X1[X1 == True].count()
        self.mean_spb = round((X1 * y).sum() / spb_count)
        d2 = {"msk": True, "spb": False}
        X2 = X["city"].map(d2)
        self.mean_msk = round((X2 * y).sum() / msk_count)
        return self

    def predict(self, X=None):
        d1 = {"msk": self.mean_msk, "spb": self.mean_spb}
        y_pred = X["city"].map(d1)
        return y_pred.to_numpy()


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        X1 = X
        X1["average_bill"] = y
        self.median_checks = X1.groupby(["city", "modified_rubrics"])[
            "average_bill"
        ].median()
        return self

    def predict(self, X=None):
        y_pred = np.full(shape=X.shape[0], fill_value=0)
        for i in range(len(y_pred)):
            y_pred[i] = float(
                self.median_checks.loc[
                    (X["city"].iloc[i], X["modified_rubrics"].iloc[i])
                ]
            )

        return y_pred


from sklearn.metrics import accuracy_score


class RubricFeaturesClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        X1 = X
        X1["average_bill"] = y
        self.median_modified_features = X1.groupby(["modified_features"])[
            "average_bill"
        ].median()
        self.median_global = y.median()
        return self

    def predict(self, X=None):
        y_pred = np.full(shape=X.shape[0], fill_value=0)
        for i in range(len(y_pred)):
            if X["modified_features"].iloc[i] in self.median_modified_features:
                y_pred[i] = int(
                    self.median_modified_features.loc[(X["modified_features"].iloc[i])]
                )
            else:
                y_pred[i] = self.median_global
        return y_pred


from catboost import CatBoostClassifier
