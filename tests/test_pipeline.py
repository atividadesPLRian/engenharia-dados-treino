import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path

# Adjust the path to import src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ingest_to_bronze, process_to_silver, model_to_gold

@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as bronze_dir, \
         tempfile.TemporaryDirectory() as silver_dir, \
         tempfile.TemporaryDirectory() as gold_dir:
        yield raw_dir, bronze_dir, silver_dir, gold_dir

@pytest.fixture
def sample_raw_data(temp_dirs):
    raw_dir, _, _, _ = temp_dirs
    raw_path = os.path.join(raw_dir, 'sales.csv')
    
    csv_content = """transaction_id,date,customer_id,customer_name,city,product_name,category,quantity,unit_price
1,2023-01-15,101,João Silva,São Paulo,Laptop,Electronics,1,$1200.50
2,15/01/2023,102,Maria Souza,rio de janeiro,Smartphone,electronics,2,R$ 800.00
3,2023-01-16,101,João Silva,sao paulo,Headphones,Electronics,,$150.00"""
    
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
        
    return raw_path

def test_ingest_to_bronze(temp_dirs, sample_raw_data):
    _, bronze_dir, _, _ = temp_dirs
    bronze_path = ingest_to_bronze(sample_raw_data, bronze_dir)
    
    assert os.path.exists(bronze_path)
    df = pd.read_parquet(bronze_path)
    assert len(df) == 3
    assert 'transaction_id' in df.columns

def test_process_to_silver(temp_dirs, sample_raw_data):
    _, bronze_dir, silver_dir, _ = temp_dirs
    bronze_path = ingest_to_bronze(sample_raw_data, bronze_dir)
    silver_path = process_to_silver(bronze_path, silver_dir)
    
    assert os.path.exists(silver_path)
    df = pd.read_parquet(silver_path)
    
    # Check if dates are parsed correctly
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    # Check if city casing is fixed
    assert set(df['city'].unique()) == {'São Paulo', 'Rio De Janeiro'}
    
    # Check if unit_price is float and clean
    assert pd.api.types.is_float_dtype(df['unit_price'])
    assert df['unit_price'].iloc[0] == 1200.50
    
    # Check if missing quantity was filled
    assert df['quantity'].iloc[2] == 1.0

def test_model_to_gold(temp_dirs, sample_raw_data):
    _, bronze_dir, silver_dir, gold_dir = temp_dirs
    bronze_path = ingest_to_bronze(sample_raw_data, bronze_dir)
    silver_path = process_to_silver(bronze_path, silver_dir)
    paths = model_to_gold(silver_path, gold_dir)
    
    dim_customer_path, dim_product_path, fact_sales_path = paths
    
    assert os.path.exists(dim_customer_path)
    assert os.path.exists(dim_product_path)
    assert os.path.exists(fact_sales_path)
    
    dim_cust = pd.read_parquet(dim_customer_path)
    dim_prod = pd.read_parquet(dim_product_path)
    fact = pd.read_parquet(fact_sales_path)
    
    assert 'customer_id' in dim_cust.columns
    assert 'product_id' in dim_prod.columns
    assert 'total_amount' in fact.columns
    assert len(fact) == 3
