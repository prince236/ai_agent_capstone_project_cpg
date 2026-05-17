import pandas as pd

def load_data(path="data/cpg_sales_sample_5000.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df
