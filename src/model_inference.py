import pandas as pd
import joblib
import argparse
from feature_engineering import create_features

def predict_next_day(csv_path, model_path="model.plk"):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = create_features(df)

    X = df.drop(["Tomorrow_Close"], axis=1, errors='ignore')
    latest_data = X.iloc[[-1]]

    model = joblib.load(model_path)

    prediction = model.predict(latest_data)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict next day stock movements using a pre-trained model."
    )
    parser.add_argument("--csv-path", type=str, default="data/stock_data.csv",
                        help="Path to the CSV file containing stock data (default: data/stock_data.csv)")
    parser.add_argument("--model-path", type=str, default="model.plk",
                        help="Path to the saved model file (default: model.plk)")
    args = parser.parse_args()

    result = predict_next_day(args.csv_path, model_path=args.model_path)

    if result == 1:
        print("The model predicts an increase in stock price.")
    else:
        print("The model predicts a decrease in stock price.")