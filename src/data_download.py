import yfinance as yf
import argparse
import pandas as pd

def download_stock_data(ticker, start_date, end_date, csv_path="data/stock_data.csv"):
    df = yf.download(ticker, start=start_date, end=end_date)
    # Reset the index so that the date becomes a column called "Date"
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if csv_path:
        # Save the DataFrame with a header that includes "Date"
        df.to_csv(csv_path, index=False)
    return df

def get_args():
    parser = argparse.ArgumentParser(prog='StockPicker',
                                     description='Download stock data using yfinance')
    parser.add_argument('--ticker', type=str, help="Stock ticker (e.g., AAPL)")
    parser.add_argument('--start-date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--csv-path', type=str, default="data/stock_data.csv",
                        help="CSV path to save stock data")
    
    args = parser.parse_args()

    if not args.ticker:
        args.ticker = input("Enter the stock ticker (e.g., AAPL): ").upper().strip()
    else:
        args.ticker = args.ticker.upper().strip()

    if not args.start_date:
        args.start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    else:
        args.start_date = args.start_date.strip()

    if not args.end_date:
        args.end_date = input("Enter the end date (YYYY-MM-DD): ").strip()
    else:
        args.end_date = args.end_date.strip()

    return args

if __name__ == "__main__":
    args = get_args()
    df = download_stock_data(args.ticker, args.start_date, args.end_date, args.csv_path)
    print(df.head())
