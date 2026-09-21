import pandas as pd
import os
from pathlib import Path

def get_project_root():
    """Returns the root path of the project."""
    return Path(__file__).parent.parent

def ingest_to_bronze(raw_file_path: str, bronze_dir: str):
    """Reads raw CSV and saves as Parquet in bronze layer."""
    os.makedirs(bronze_dir, exist_ok=True)
    df = pd.read_csv(raw_file_path)
    
    file_name = Path(raw_file_path).stem
    bronze_path = os.path.join(bronze_dir, f"{file_name}.parquet")
    
    df.to_parquet(bronze_path, index=False)
    print(f"Bronze data saved to {bronze_path}")
    return bronze_path

def process_to_silver(bronze_file_path: str, silver_dir: str):
    """Cleans data from bronze and saves to silver layer."""
    os.makedirs(silver_dir, exist_ok=True)
    df = pd.read_parquet(bronze_file_path)
    
    # 1. Clean dates: convert to datetime, coercing errors to NaT
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
    
    # 2. Clean text casing
    df['city'] = df['city'].str.title().str.strip()
    # Normalize 'Sao Paulo' vs 'São Paulo'
    df['city'] = df['city'].replace({'Sao Paulo': 'São Paulo'})
    
    df['category'] = df['category'].str.title().str.strip()
    
    # 3. Clean unit_price: remove currency symbols and convert to float
    if df['unit_price'].dtype == 'O': # Object / String
        df['unit_price'] = df['unit_price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        
    # 4. Handle missing or negative quantity
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1)
    df.loc[df['quantity'] < 0, 'quantity'] = 1
    
    # 5. Drop rows where critical fields are null (e.g., date)
    df = df.dropna(subset=['date', 'customer_id'])
    
    file_name = Path(bronze_file_path).stem
    silver_path = os.path.join(silver_dir, f"{file_name}_cleaned.parquet")
    
    df.to_parquet(silver_path, index=False)
    print(f"Silver data saved to {silver_path}")
    return silver_path

def model_to_gold(silver_file_path: str, gold_dir: str):
    """Creates fact and dimension tables from silver and saves to gold layer."""
    os.makedirs(gold_dir, exist_ok=True)
    df = pd.read_parquet(silver_file_path)
    
    # Dim Customer
    dim_customer = df[['customer_id', 'customer_name', 'city']].drop_duplicates().reset_index(drop=True)
    
    # Dim Product (Creating a surrogate key as there is no product_id)
    dim_product = df[['product_name', 'category']].drop_duplicates().reset_index(drop=True)
    dim_product['product_id'] = dim_product.index + 1
    
    # Merge product_id back to main df to create Fact Sales
    df_merged = df.merge(dim_product, on=['product_name', 'category'], how='left')
    
    # Fact Sales
    fact_sales = df_merged[['transaction_id', 'date', 'customer_id', 'product_id', 'quantity', 'unit_price']].copy()
    fact_sales.loc[:, 'total_amount'] = fact_sales['quantity'] * fact_sales['unit_price']
    
    # Save to gold
    dim_customer_path = os.path.join(gold_dir, "dim_customer.parquet")
    dim_product_path = os.path.join(gold_dir, "dim_product.parquet")
    fact_sales_path = os.path.join(gold_dir, "fact_sales.parquet")
    
    dim_customer.to_parquet(dim_customer_path, index=False)
    dim_product.to_parquet(dim_product_path, index=False)
    fact_sales.to_parquet(fact_sales_path, index=False)
    
    print(f"Gold data saved to {gold_dir}")
    return dim_customer_path, dim_product_path, fact_sales_path

if __name__ == "__main__":
    root = get_project_root()
    raw_path = os.path.join(root, 'data', 'raw', 'sales.csv')
    bronze_dir = os.path.join(root, 'data', 'bronze')
    silver_dir = os.path.join(root, 'data', 'silver')
    gold_dir = os.path.join(root, 'data', 'gold')
    
    # Run pipeline
    if os.path.exists(raw_path):
        bronze_path = ingest_to_bronze(raw_path, bronze_dir)
        silver_path = process_to_silver(bronze_path, silver_dir)
        model_to_gold(silver_path, gold_dir)
    else:
        print(f"Raw file not found: {raw_path}")
