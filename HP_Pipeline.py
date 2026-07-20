# Helper Functions
from utils import *


def main():
    df = load_data()
    X, y = houseprice_data_prep(df)
    base_models_regression(X, y)
    best_models = hyperparameter_optimization_regression(X, y, cv=3)
    voting_reg = voting_regressor(best_models, X, y)
    joblib.dump(voting_reg, "voting_reg.pkl")
    return voting_reg


if __name__ == "__main__":
    main()
