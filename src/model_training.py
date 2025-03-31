import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse

from feature_engineering import create_features, create_target

def train_with_gridsearch(csv_path, model_path="model.plk", n_splits=5):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    df = create_features(df)
    df = create_target(df)

    X = df.drop(["Tomorrow_Close", "Target"], axis=1)
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_grid = {
        'n_estimators': list(range(10, 201, 10)),
        'max_depth': [None] + list(range(1, 30, 2))
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='accuracy')
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X)
    print("Accuracy on full data:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    joblib.dump(best_model, model_path)
    print("Final model saved to", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a RandomForest model for stock movement prediction."
    )
    parser.add_argument("--csv-path", type=str, default="data/stock_data.csv",
                        help="Path to the CSV file containing stock data (default: data/stock_data.csv)")
    parser.add_argument("--model-path", type=str, default="model.plk",
                        help="Path to save the trained model (default: model.plk)")
    args = parser.parse_args()
    
    train_with_gridsearch(args.csv_path, model_path=args.model_path, n_splits=5)

    
