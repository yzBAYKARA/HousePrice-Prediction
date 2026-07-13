import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Scaling
from sklearn.preprocessing import RobustScaler

# Linear Models
from sklearn.linear_model import LogisticRegression, LinearRegression

# Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# SVM
from sklearn.svm import SVC

# Trees
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Ensemble
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    VotingClassifier,
    VotingRegressor,
)

# Model Selection & Preprocessing
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Boosting
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


# =============================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def check_df(dataframe, head=5):
    """Summarizes the overall structure of a dataset,
     including its shape, data types, first/last rows, missing values, and numerical quantiles."""

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical-but-cardinal variables in a dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
    cat_th : int
        Threshold for treating numerical variables as categorical based on the number of unique values.
    car_th : int
        Threshold for treating categorical variables as cardinal based on the number of unique values.

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numerical variables.
    cat_but_car : list
        List of categorical variables with high cardinality.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]
    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtypes != "O" and col not in num_but_cat
    ]
    return cat_cols, num_cols, cat_but_car


def categoric_summary(dataframe, col_name, plot=False):
    """Displays the frequency table and percentage distribution of a categorical variable."""
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "ratio": 100 * dataframe[col_name].value_counts() / len(dataframe[col_name]),
    }))
    print("#########################################")

    if plot:
        dataframe[col_name].value_counts().plot(kind="bar")
        plt.title(col_name)
        plt.show(block=True)


def numeric_summary(dataframe, col_name, plot=False):
    """Displays descriptive statistics for a numerical variable and optionally plots its histogram."""
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    """Displays the mean of a numerical variable grouped by the target variable."""
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    """Displays the mean of the target variable grouped by a categorical variable."""
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


def correlation_matrix(df, cols):
    """Plots a correlation heatmap for the selected columns."""
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    sns.heatmap(
        df[cols].corr(), annot=True, linewidths=0.5,
        annot_kws={"size": 12}, linecolor="w", cmap="RdBu",
    )
    plt.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    Identifies highly correlated columns based on the specified correlation threshold.

    Returns
    -------
    trash_list : list
        List of column names recommended for removal due to high correlation.
    """
    numeric_list = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    corr_matrix = dataframe[numeric_list].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    trash_list = [col for col in upper if any(upper[col] > corr_th)]

    if plot:
        corr = dataframe[numeric_list].drop(trash_list, axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(
            corr, mask=mask, cmap="RdBu", annot=True, fmt=".2f",
            vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax,
        )
        ax.set_title("Correlation Matrix", fontsize=16, pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show(block=True)

    return trash_list


# =============================================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

# --- Outlier ---

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """Calculates the lower and upper outlier thresholds using the IQR method."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    """Checks whether a column contains outliers and returns True or False."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def grab_outliers(dataframe, col_name, index=False):
    """
    Displays the outliers in a column. If index=True, returns the indices of the outlier rows.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outlier_df = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]

    if outlier_df.shape[0] > 10:
        print(outlier_df.head())
    else:
        print(outlier_df)

    if index:
        return outlier_df.index


def remove_outlier(dataframe, col_name):
    """Removes outliers from the dataset and returns the filtered DataFrame."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]


def replace_with_thresholds(dataframe, col_name):
    """Caps outliers at the calculated lower and upper threshold values (in-place)."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit


# --- Missing Values ---

def missing_values_table(dataframe, na_name=False):
    """
    Displays the number and percentage of missing values for each column.
    If na_name=True, returns the list of columns containing missing values.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    nmiss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([nmiss, np.round(ratio, 2)], axis=1, keys=["nmiss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """
    Creates missing value indicator (NA flag) variables for the specified columns
    and displays the mean of the target variable grouped by each NA flag.
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.columns[temp_df.columns.str.contains("_NA_")]
    for col in na_flags:
        print(pd.DataFrame({
            "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
            "Count": temp_df.groupby(col)[target].count(),
        }), end="\n\n\n")


# --- Encoding ---

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """Applies one-hot encoding to the specified categorical columns."""
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)


def label_encoder(dataframe, binary_col):
    """Applies label encoding to a binary (two-class) categorical column."""
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


# --- Rare Encoding ---

def rare_analyser(dataframe, target, cat_cols):
    """Analyzes categorical variables by displaying category counts,
     ratios, and the mean of the target variable for each category."""

    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean(),
        }).sort_values("RATIO", ascending=False), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    """
    Groups categories with frequencies below the specified threshold into a single 'Rare' category.

    Parameters
    ----------
    rare_perc : float
        Frequency threshold below which categories are considered rare (e.g., 0.01).
    """
    temp_df = dataframe.copy()
    rare_columns = [
        col for col in temp_df.columns
        if temp_df[col].dtypes == "O"
        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)
    ]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df


# =============================================================================
# 3. BASE MODELS
# =============================================================================

base_regressors = [
    ("LR", LinearRegression()),
    ("KNN", KNeighborsRegressor()),
    ("CART", DecisionTreeRegressor(random_state=42)),
    ("RF", RandomForestRegressor(random_state=42)),
    ("Adaboost", AdaBoostRegressor(random_state=42)),
    ("GBM", GradientBoostingRegressor(random_state=42)),
    ("XGBoost", XGBRegressor(random_state=42)),
    ("LightGBM", LGBMRegressor(verbose=-1, random_state=42)),
]
def base_models_regression(X, y, scoring="neg_mean_absolute_error"):
    """Evaluates multiple regression models using cross-validation and prints their baseline performance scores."""
    print("Base Models......")
    for name, regressor in base_regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")


def base_models_classification(X, y, scoring="roc_auc"):
    """Evaluates multiple classification models using cross-validation and prints their baseline performance scores."""
    print("Base Models......")
    classifiers = [
        ("LR", LogisticRegression()),
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ("Adaboost", AdaBoostClassifier()),
        ("GBM", GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier(eval_metric="logloss")),
        ("LightGBM", LGBMClassifier(verbose=-1)),
        # ("Catboost", CatBoostClassifier(verbose=False)),
    ]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")


# =============================================================================
# 4. HYPERPARAMETER OPTIMIZATION
# =============================================================================

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {
    "max_depth": range(1, 20),
    "min_samples_split": range(2, 30),
}

rf_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
}

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 200],
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
}


classifiers = [
    ("KNN", KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ("XGBoost", XGBClassifier(eval_metric="logloss", device="cpu"), xgboost_params),
    ("LightGBM", LGBMClassifier(verbose=-1), lightgbm_params),
]

def hyperparameter_optimization_classification(X, y, cv=3, scoring="roc_auc"):
    """
    Performs hyperparameter tuning for the predefined classification models using GridSearchCV.

    Returns
    -------
    best_models : dict
        Dictionary containing the optimized models in the format {model_name: optimized_model}.
    """
    print("Hyperparameter Optimization.....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"######## {name} ########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models

regressors = [
    ("CART", DecisionTreeRegressor(random_state=42), {'max_depth': [None] + list(range(1, 35)),
                                       "min_samples_split": range(2, 15)}),
    ("RF", RandomForestRegressor(random_state=42), {"max_depth": [None, 10, 20, 30],
                                     "max_features": ["sqrt", "log2", 1.0],
                                     "min_samples_split": [2, 5, 10],
                                    "min_samples_leaf": [1, 2, 4],
                                     "n_estimators": [100, 200, 300, 500]}),
    ("XGBoost", XGBRegressor(device='cpu'), {"learning_rate": [0.1, 0.01],
                                  "max_depth": [5, 8],
                                  "n_estimators": [100, 200]}),
    ("LightGBM", LGBMRegressor(verbose=-1, random_state=42), {"learning_rate": [0.01, 0.1, 0.5],
                                              "n_estimators": list(range(100, 1000, 100))}),
    ("GBM", GradientBoostingRegressor(random_state=42), {"learning_rate": [0.1, 0.01, 0.5],
                                           "max_depth": range(1, 6),
                                           "n_estimators": range(100, 1000, 100)}),
]
def hyperparameter_optimization_regression(X, y, cv=5, scoring="neg_mean_absolute_error"):
    """
    Performs hyperparameter tuning for the predefined regression models using GridSearchCV.

    Returns
    -------
    best_models : dict
        Dictionary containing the optimized models in the format {model_name: optimized_model}.
    """
    print("Hyperparameter Optimization.....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"######## {name} ########")
        cv_results = cross_validate(regressor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models



# =============================================================================
# 5. ENSEMBLE / STACKING
# =============================================================================

def voting_classifier(best_models, X, y):
    """
    Combines the KNN, Random Forest, and LightGBM models using soft voting.

    Returns
    -------
    voting_clf : fitted VotingClassifier
        Trained VotingClassifier ensemble model.
    """
    print("Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ("KNN", best_models["KNN"]),
            ("RF", best_models["RF"]),
            ("LightGBM", best_models["LightGBM"]),
        ],
        voting="soft",
    ).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"F1 Score : {cv_results['test_f1'].mean():.4f}")
    print(f"ROC AUC  : {cv_results['test_roc_auc'].mean():.4f}")
    return voting_clf

def voting_regressor(best_models, X, y):
    """
    Combines the GBM, XGBoost, and LightGBM models using a Voting Regressor.

    Returns
    -------
    voting_reg : fitted VotingRegressor
        Trained VotingRegressor ensemble model.
    """
    print("Voting Regressor...")
    voting_reg = VotingRegressor(
        estimators=[
            ("GBM", best_models["GBM"]),
            ("XGBoost", best_models["XGBoost"]),
            ("LightGBM", best_models["LightGBM"]),
        ]
    ).fit(X, y)

    cv_results = cross_validate(
        voting_reg, X, y, cv=5,
        scoring=["neg_mean_absolute_error", "neg_mean_squared_error", "r2"],
    )
    print(f"MAE : {-cv_results['test_neg_mean_absolute_error'].mean():.4f}")
    print(f"MSE : {-cv_results['test_neg_mean_squared_error'].mean():.4f}")
    print(f"R2  : {cv_results['test_r2'].mean():.4f}")
    return voting_reg

def houseprice_data_prep(dataframe):
        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
        num_cols.remove("MSSubClass")
        num_cols.remove("MoSold")
        cat_cols.append("MSSubClass")
        cat_cols.append("MoSold")

        # Eksik değer doldurma
        dataframe["LotFrontage"] = dataframe["LotFrontage"].fillna(dataframe["LotFrontage"].median())
        dataframe["MasVnrArea"] = dataframe["MasVnrArea"].fillna(dataframe["MasVnrArea"].median())
        dataframe["Electrical"] = dataframe["Electrical"].fillna(dataframe["Electrical"].mode()[0])

        dataframe["Alley"] = dataframe["Alley"].fillna("None")
        dataframe["MasVnrType"] = dataframe["MasVnrType"].fillna("None")
        dataframe["FireplaceQu"] = dataframe["FireplaceQu"].fillna("None")
        dataframe["Fence"] = dataframe["Fence"].fillna("None")
        dataframe["MiscFeature"] = dataframe["MiscFeature"].fillna("None")

        dataframe['HAS_POOL'] = dataframe['PoolQC'].notnull().astype(int)
        dataframe['PoolQC'] = dataframe['PoolQC'].fillna("None")

        garage_cols = [col for col in dataframe.columns if "Garage" in col]
        bsmt_cols = [col for col in dataframe.columns if "Bsmt" in col]

        for col in garage_cols:
            if dataframe[col].dtypes == 'object':
                dataframe[col] = dataframe[col].fillna("None")
            else:
                dataframe[col] = dataframe[col].fillna(0)

        for col in bsmt_cols:
            if dataframe[col].dtypes == 'object':
                dataframe[col] = dataframe[col].fillna("None")
            else:
                dataframe[col] = dataframe[col].fillna(0)

        quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        bsmt_fin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

        bsmt_fin_cols = ['BsmtFinType1', 'BsmtFinType2']
        for col in bsmt_fin_cols:
            dataframe[col] = dataframe[col].map(bsmt_fin_map)

        other_quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                              'HeatingQC', 'KitchenQual', 'FireplaceQu',
                              'GarageQual', 'GarageCond', 'PoolQC']
        for col in other_quality_cols:
            dataframe[col] = dataframe[col].map(quality_map)

        dataframe["QualityScore"] = dataframe[other_quality_cols].sum(axis=1) + dataframe[bsmt_fin_cols].sum(axis=1) + \
                                    dataframe['OverallQual'] + dataframe[
                                        'OverallCond']
        dataframe["Total_Bath"] = dataframe["BsmtFullBath"] + dataframe["BsmtHalfBath"] + dataframe["FullBath"] + \
                                  dataframe["HalfBath"]
        dataframe['IS_RENOVATED'] = (dataframe['YearRemodAdd'] != dataframe['YearBuilt']).astype(int)
        dataframe['YearsSinceRemodel'] = dataframe['YrSold'] - dataframe['YearRemodAdd']
        dataframe.loc[dataframe['YearsSinceRemodel'] < 0, 'YearsSinceRemodel'] = 0

        # grab_col_names güncelle
        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
        num_cols.remove("MSSubClass")
        num_cols.remove("MoSold")
        cat_cols.append("MSSubClass")
        cat_cols.append("MoSold")

        # Outlier
        discrete_cols = ['OverallQual', 'TotRmsAbvGrd', '3SsnPorch', 'LowQualFinSF']
        year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
        exclude = ['Id', 'SalePrice'] + discrete_cols + year_cols
        continuous_cols = [col for col in num_cols if col not in exclude]

        for col in continuous_cols:
            replace_with_thresholds(dataframe, col)

        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
        num_cols.remove("MSSubClass")
        num_cols.remove("MoSold")
        cat_cols.append("MSSubClass")
        cat_cols.append("MoSold")

        dataframe = rare_encoder(dataframe, 0.01)

        # Label Encode
        dataframe = label_encoder(dataframe, "CentralAir")

        # One Hot Encode
        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, car_th=30)
        ohe_cols = [col for col in cat_cols if dataframe[col].dtype == 'object']
        dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)

        # Scaling
        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
        scaler = RobustScaler()
        scale_cols = [col for col in num_cols if col not in ['SalePrice', 'Id']]
        dataframe[scale_cols] = scaler.fit_transform(dataframe[scale_cols])

        y = dataframe["SalePrice"]
        X = dataframe.drop(["SalePrice"], axis=1)
        return X, y
