######################################
# Prediction
######################################

from utils import *


def main():
    df = load_data()

    X, y = houseprice_data_prep(df)

    random_house = X.sample(1, random_state=42)

    model = joblib.load("voting_reg.pkl")

    prediction = model.predict(random_house)

    print(f"Predicted Sale Price: {prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
