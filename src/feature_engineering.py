import pandas as pd

def create_features(df):
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
    df["Daily_return"] = df["Close"].pct_change()
    df["Range"] = df["High"] - df["Low"]
    df.dropna(inplace=True)
    return df

def create_target(df):
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)
    return df

