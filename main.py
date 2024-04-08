import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import ml
from sklearn.model_selection import train_test_split


# --------1---------
def construct_matrix(first_array, second_array):
    return np.hstack(
        [
            np.reshape(first_array, (first_array.size, 1)),
            np.reshape(second_array, (second_array.size, 1)),
        ]
    )


# --------2---------
def most_frequent(nums):
    counts = np.bincount(nums)
    return np.argmax(counts)


# --------Работа с данными---------
data = pd.read_csv("organisations.csv")
features = pd.read_csv("features.csv")
rubrics = pd.read_csv("rubrics.csv")

r_id = rubrics["rubric_id"]
r_name = rubrics["rubric_name"]
rubric_dict = dict(zip(r_name, r_id))
f_id = features["feature_id"]
f_name = features["feature_name"]
features_dict = dict(zip(f_name, f_id))


low_border = 0
upper_border = 5000

x1 = list(
    data[data["city"] == "msk"][
        (data["average_bill"] >= low_border) & (data["average_bill"] <= upper_border)
    ]["average_bill"]
)
x2 = list(
    data[data["city"] == "spb"][
        (data["average_bill"] >= low_border) & (data["average_bill"] <= upper_border)
    ]["average_bill"]
)
names = ["msk", "spb"]
plt.hist([x1, x2], bins=int(10), stacked=True, label=names)
plt.show()

# --------3---------
data = data[(data["average_bill"] <= 2500) & (data["average_bill"] >= 0)]

# --------4---------
grouped = data.groupby("city")["average_bill"].mean()
print(round(grouped["msk"] - grouped["spb"]))

# --------Применяем ML---------
clean_data_train, clean_data_test = train_test_split(
    data, stratify=data["average_bill"], test_size=0.33, random_state=42
)
# --------5 и 6--------- Модели в файле mp.py
reg = ml.MeanRegressor()
reg.fit(y=clean_data_train["average_bill"])
y_pred_reg = reg.predict(X=clean_data_test["features_id"].values)
rmse_reg = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_pred_reg)
)

clf = ml.MostFrequentClassifier()
clf.fit(y=clean_data_train["average_bill"])
y_pred_clf = clf.predict(X=clean_data_test["features_id"].values)
rmse_clf = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_pred_clf)
)
score_clf = ml.balanced_accuracy_score(
    clean_data_test["average_bill"].values, y_pred_clf
)

# --------7---------
city_reg = ml.CityMeanRegressor()
city_reg.fit(X=clean_data_train, y=clean_data_train["average_bill"])
y_pred_city_reg = city_reg.predict(X=clean_data_test)
rmse_city_reg = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_pred_city_reg)
)

# --------8---------
clean_data_train["modified_rubrics"] = clean_data_train["rubrics_id"]
clean_data_test["modified_rubrics"] = clean_data_test["rubrics_id"]

counts = Counter(clean_data_train["rubrics_id"])
clean_data_train["modified_rubrics"] = clean_data_train["modified_rubrics"].apply(
    lambda x: "other" if counts[x] < 100 else x
)
clean_data_test["modified_rubrics"] = clean_data_test["modified_rubrics"].apply(
    lambda x: "other" if counts.get(x, 0) < 100 else x
)
# --------ИЛИ---------
""" replacements = clean_data_train["rubrics_id"].value_counts() 
replacements = replacements[replacements < 100].index.tolist()
clean_data_train["modified_rubrics"] = clean_data_train["modified_rubrics"].apply(
    lambda x: "other" if x in replacements else x
)
clean_data_test["modified_rubrics"] = clean_data_train["modified_rubrics"].apply(
    lambda x: "other" if x in replacements else x
)  """

rub_city_clf = ml.RubricCityMedianClassifier()
rub_city_clf.fit(X=clean_data_train, y=clean_data_train["average_bill"])
y_pred_rub_city_clf = rub_city_clf.predict(X=clean_data_test)

rmse_rub_city_clf = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_pred_rub_city_clf)
)
score_rub_city_clf = ml.balanced_accuracy_score(
    clean_data_test["average_bill"].values, y_pred_rub_city_clf
)

# --------9---------
clean_data_train["modified_features"] = (
    clean_data_train["rubrics_id"].astype(str)
    + "q"
    + clean_data_train["features_id"].astype(str)
)
clean_data_test["modified_features"] = (
    clean_data_test["rubrics_id"].astype(str)
    + "q"
    + clean_data_test["features_id"].astype(str)
)

replacements = clean_data_train["modified_features"].unique()
clean_data_test["modified_features"] = clean_data_test["modified_features"].apply(
    lambda x: x if x in replacements else "other"
)
rub_feu_clf = ml.RubricFeaturesClassifier()
rub_feu_clf.fit(X=clean_data_train, y=clean_data_train["average_bill"])
y_pred_rub_feu_clf_tr = rub_feu_clf.predict(X=clean_data_train)
y_pred_rub_feu_clf = rub_feu_clf.predict(X=clean_data_test)

rmse0 = np.sqrt(
    ml.mean_squared_error(
        clean_data_train["average_bill"].values, y_pred_rub_feu_clf_tr
    )
)
score0 = ml.balanced_accuracy_score(
    clean_data_train["average_bill"].values, y_pred_rub_feu_clf_tr
)

rmse1 = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_pred_rub_feu_clf)
)
score1 = ml.balanced_accuracy_score(
    clean_data_test["average_bill"].values, y_pred_rub_feu_clf
)

frame_predict = clean_data_test
frame_predict["predict"] = y_pred_rub_feu_clf.tolist()
frame_predict.drop(
    [
        "org_id",
        "city",
        "features_id",
        "modified_features",
        "average_bill",
        "rating",
        "rubrics_id",
    ],
    axis=1,
    inplace=True,
)
frame_predict.to_csv("my_data.csv", header=False, index=True)

# --------10---------
sparse_data_train = clean_data_train.copy()
sparse_data_train.drop(
    ["org_id", "features_id", "rubrics_id", "average_bill"],
    axis=1,
    inplace=True,
)
d = {"msk": 1, "spb": 0}
sparse_data_train["city"] = sparse_data_train["city"].map(d)
df_dummies_r = clean_data_train["rubrics_id"].str.get_dummies(sep=" ")
df_dummies_f = clean_data_train["features_id"].str.get_dummies(sep=" ")
sparse_data_train = pd.concat([sparse_data_train, df_dummies_r], axis=1)
sparse_data_train["feature_other"] = 0
sparse_data_train = pd.concat([sparse_data_train, df_dummies_f], axis=1)


sparse_data_test = clean_data_test.copy()
sparse_data_test.drop(
    ["org_id", "features_id", "rubrics_id", "average_bill"],
    axis=1,
    inplace=True,
)
d = {"msk": 1, "spb": 0}
sparse_data_test["city"] = sparse_data_test["city"].map(d)
df_dummies_r1 = clean_data_test["rubrics_id"].str.get_dummies(sep=" ")
sparse_data_test = pd.concat([sparse_data_test, df_dummies_r1], axis=1)
df_dummies_f1 = clean_data_test["features_id"].str.get_dummies(sep=" ")
missing_columns = set(df_dummies_f1.columns) - set(df_dummies_f.columns)
sparse_data_test["feature_other"] = 0
for col in missing_columns:
    sparse_data_test["feature_other"] += df_dummies_f1[col]
    df_dummies_f1.drop([col], axis=1, inplace=True)
sparse_data_test = pd.concat([sparse_data_test, df_dummies_f1], axis=1)
sparse_data_test = sparse_data_test.reindex(columns=sparse_data_train.columns)

clf = ml.CatBoostClassifier()
clf.fit(sparse_data_train, clean_data_train["average_bill"])
y_catboost_predict = clf.predict(sparse_data_test)
rmse = np.sqrt(
    ml.mean_squared_error(clean_data_test["average_bill"].values, y_catboost_predict)
)
print(rmse)
