########################################################################
# House Price Project
########################################################################

from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv(r"C:\Users\yusuf\PycharmProjects\miuul_course\datasets\housepricetrain.csv")

check_df(df)

def update_cols(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=10)
    num_cols.remove("MSSubClass")
    num_cols.remove("MoSold")
    cat_cols.append("MSSubClass")
    cat_cols.append("MoSold")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = update_cols(df)

for col in cat_cols:
    categoric_summary(df, col)

df[num_cols].describe().T

for col in num_cols:
    numeric_summary(df, col)

na_cols = missing_values_table(df, True)
missing_vs_target(df, "SalePrice", na_cols)

rare_analyser(df, 'SalePrice', cat_cols)

for col in num_cols:
    print(check_outlier(df, col, q1=0.10, q3=0.90))

correlation_matrix(df, num_cols)
high_correlated_cols(df, True)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Missing Values
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"]  = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["Electrical"]  = df["Electrical"].fillna(df["Electrical"].mode()[0])

df["Alley"]       = df["Alley"].fillna("None")
df["MasVnrType"]  = df["MasVnrType"].fillna("None")
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df["Fence"]       = df["Fence"].fillna("None")
df["MiscFeature"] = df["MiscFeature"].fillna("None")

df['HAS_POOL'] = df['PoolQC'].notnull().astype(int)
df['PoolQC']   = df['PoolQC'].fillna("None")

garage_cols = [col for col in df.columns if "Garage" in col]
bsmt_cols   = [col for col in df.columns if "Bsmt" in col]

for col in garage_cols:
    if df[col].dtypes == 'object':
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)

for col in bsmt_cols:
    if df[col].dtypes == 'object':
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)

# Feature Extractions
quality_map  = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_fin_map = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

bsmt_fin_cols = ['BsmtFinType1', 'BsmtFinType2']
for col in bsmt_fin_cols:
    df[col] = df[col].map(bsmt_fin_map)

other_quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                      'HeatingQC', 'KitchenQual', 'FireplaceQu',
                      'GarageQual', 'GarageCond', 'PoolQC']
for col in other_quality_cols:
    df[col] = df[col].map(quality_map)

df["QualityScore"]    = df[other_quality_cols].sum(axis=1) + df[bsmt_fin_cols].sum(axis=1) + df['OverallQual'] + df['OverallCond']
df["Total_Bath"]      = df["BsmtFullBath"] + df["BsmtHalfBath"] + df["FullBath"] + df["HalfBath"]
df['IS_RENOVATED']    = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
df.loc[df['YearsSinceRemodel'] < 0, 'YearsSinceRemodel'] = 0

cat_cols, num_cols, cat_but_car = update_cols(df)

# Outlier
discrete_cols = ['OverallQual', 'TotRmsAbvGrd', '3SsnPorch', 'LowQualFinSF']
year_cols     = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
exclude       = ['Id', 'SalePrice'] + discrete_cols + year_cols
continuous_cols = [col for col in num_cols if col not in exclude]

for col in continuous_cols:
    replace_with_thresholds(df, col)

# Rare Analysis
cols_to_drop  = ["Street", "Utilities", "RoofMatl", "Condition2"]
df.drop(cols_to_drop, axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = update_cols(df)

temp_cat_cols = [col for col in cat_cols if df[col].dtype == 'object']
rare_analyser(df, "SalePrice", temp_cat_cols)

df = rare_encoder(df, 0.01)

# Label Encode
df = label_encoder(df, "CentralAir")

# One Hot Encode
cat_cols, num_cols, cat_but_car = grab_col_names(df, car_th=30)
ohe_cols = [col for col in cat_cols if df[col].dtype == 'object']
df = one_hot_encoder(df, ohe_cols, drop_first=True)

# Scaling
cat_cols, num_cols, cat_but_car = update_cols(df)
scaler     = RobustScaler()
scale_cols = [col for col in num_cols if col not in ['SalePrice', 'Id']]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Model
df.drop("Id", axis=1, inplace=True)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Base Models
base_models_regression(X, y)

