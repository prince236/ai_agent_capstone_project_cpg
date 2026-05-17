import pandas as pd
import numpy as np
import os

def generate_cpg_data(output_path="data/cpg_sales_sample_5000.csv", num_rows=5000):
    np.random.seed(123)

    # Parameters
    date_range = pd.date_range('2022-01-01', periods=365)
    stores = list(range(1, 11))
    regions = ['North', 'South', 'East']
    skus = list(range(101, 151))
    categories = ['Beverages', 'Snacks', 'Dairy', 'Household', 'Personal Care']
    promo_types = ['Discount', 'BuyOneGetOne', 'FlashSale']
    store_sizes = ['Small', 'Medium', 'Large']

    data = []
    for _ in range(num_rows):
        date = np.random.choice(date_range)
        store = np.random.choice(stores)
        store_region = np.random.choice(regions)
        sku = np.random.choice(skus)
        category = np.random.choice(categories)
        base_price = np.round(np.random.uniform(2, 10), 2)
        promo_flag = np.random.choice([0, 1], p=[0.8, 0.2])
        promo_type = np.random.choice(promo_types) if promo_flag else None
        price = np.round(base_price * (0.8 if promo_flag else 1.0), 2)
        units_sold = np.random.poisson(20) if promo_flag else np.random.poisson(10)
        revenue = np.round(units_sold * price, 2)
        inventory_level = np.random.randint(100, 1000)
        store_size = np.random.choice(store_sizes)
        holiday_flag = 1 if pd.to_datetime(date).dayofweek in [5, 6] else 0

        data.append([date, store, store_region, sku, category,
                     units_sold, revenue, promo_flag, promo_type,
                     price, inventory_level, store_size
                     ,holiday_flag
                    ])

    df = pd.DataFrame(data, columns=[
        'date', 'store_id', 'store_region', 'sku_id', 'category',
        'units_sold', 'revenue', 'promo_flag', 'promo_type',
        'price', 'inventory_level', 'store_size', 'holiday_flag'
    ])

    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print("CSV with 5000 rows created successfully.")

if __name__ == "__main__":
    generate_cpg_data()
