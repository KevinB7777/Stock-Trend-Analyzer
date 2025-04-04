import pandas as pd

def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return pd.DataFrame({
        "BB_Middle": rolling_mean,
        "BB_Upper": upper_band,
        "BB_Lower": lower_band
    })

def calculate_stochastic_oscillator(df, window=14):
    lowest_low = df["Low"].rolling(window=window).min()
    highest_high = df["High"].rolling(window=window).max()
    stoch_k = 100 * ((df["Close"] - lowest_low) / (highest_high - lowest_low))
    return stoch_k

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi
def create_features(df):
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
    df["Daily_return"] = df["Close"].pct_change()
    df["Range"] = df["High"] - df["Low"]

    df["RSI_14"] = calculate_rsi(df["Close"], period=14)
    
    bb = calculate_bollinger_bands(df["Close"], window=20, num_std=2)
    df = pd.concat([df, bb], axis=1)
    
    df["Stoch_%K"] = calculate_stochastic_oscillator(df, window=14)
    
    df["OBV"] = calculate_obv(df)

    df.dropna(inplace=True)
    return df

def create_target(df):
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)
    return df

