import pandas as pd
import numpy as np
from datetime import datetime

def create_custom_financial_dataset_with_sentiment_base(first_df_path, sentiment_df_path, output_path):
    """
    Create a custom dataset using the sentiment dataset as the base and adding
    only the moving average features from the first dataset.
    
    Parameters:
    -----------
    first_df_path : str
        Path to the first dataset file with moving averages
    sentiment_df_path : str
        Path to the sentiment dataset file
    output_path : str
        Path to save the output file
    
    Returns:
    --------
    pandas.DataFrame
        The merged dataset
    """
    print("Loading datasets...")
    
    # Load the first dataset with moving averages
    df_with_ma = pd.read_csv(first_df_path)
    print(f"Dataset with moving averages loaded: {df_with_ma.shape[0]} rows, {df_with_ma.shape[1]} columns")
    
    # Load the sentiment dataset
    sentiment_df = pd.read_csv(sentiment_df_path)
    print(f"Sentiment dataset loaded: {sentiment_df.shape[0]} rows, {sentiment_df.shape[1]} columns")
    
    # We only want these specific columns from the first dataset
    ma_columns = ['cusip', 'MthCalDt', 'SMA_3', 'EMA_3', 'SMA_6', 'EMA_6', 'SMA_12', 'EMA_12', 'LOWESS']
    sentiment_columns = ['cusip', 'sivy', 'MthCalDt', 'MthRet', 'sprtrn', 'gvkey', 'epspxy', 'epsfiy', 'oiadpy', 'niy', 'saley', 'revty', 'capxy', 'dltisy', 'unified_sentiment']

    # Check which columns exist in the first dataset
    existing_ma_columns = [col for col in ma_columns if col in df_with_ma.columns]
    missing_ma_columns = [col for col in ma_columns if col not in df_with_ma.columns]

    existing_sentiment_columns = [col for col in sentiment_columns if col in sentiment_df.columns]
    missing_sentiment_columns = [col for col in sentiment_columns if col not in sentiment_df.columns]
    
    if missing_ma_columns:
        print(f"Warning: The following moving average columns were not found: {missing_ma_columns}")
    
    if missing_sentiment_columns:
        print(f"Warning: The following moving average columns were not found: {missing_sentiment_columns}")
    
    # Select only existing moving average columns from the first dataset
    df_ma_selected = df_with_ma[existing_ma_columns].copy()
    print(f"Selected {df_ma_selected.shape[1]} columns from the moving averages dataset")
    
    # Keep all columns from the sentiment dataset
    # We'll use this as our base dataset
    # sentiment_selected = sentiment_df[existing_sentiment_columns].copy()
    sentiment_selected = sentiment_df.copy()

    print(f"Using all {sentiment_selected.shape[1]} columns from the sentiment dataset as base")
    
    # Convert date columns to datetime
    print("Converting date columns to datetime...")
    
    try:
        df_ma_selected['MthCalDt'] = pd.to_datetime(df_ma_selected['MthCalDt'])
    except Exception as e:
        print(f"Error converting MthCalDt in moving averages dataset: {e}")
        print("Attempting alternative date formats...")
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
            try:
                df_ma_selected['MthCalDt'] = pd.to_datetime(df_ma_selected['MthCalDt'], format=fmt)
                print(f"Successfully converted using format: {fmt}")
                break
            except:
                continue
    
    try:
        sentiment_selected['MthCalDt'] = pd.to_datetime(sentiment_selected['MthCalDt'])
    except Exception as e:
        print(f"Error converting MthCalDt in sentiment dataset: {e}")
        print("Attempting alternative date formats...")
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
            try:
                sentiment_selected['MthCalDt'] = pd.to_datetime(sentiment_selected['MthCalDt'], format=fmt)
                print(f"Successfully converted using format: {fmt}")
                break
            except:
                continue
    
    # Standardize CUSIP format in both datasets
    print("Standardizing CUSIP format...")
    df_ma_selected['cusip'] = df_ma_selected['cusip'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str[:8]
    sentiment_selected['cusip'] = sentiment_selected['cusip'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str[:8]
    
    # Check for duplicates in the merge keys
    ma_dups = df_ma_selected.duplicated(subset=['cusip', 'MthCalDt']).sum()
    sentiment_dups = sentiment_selected.duplicated(subset=['cusip', 'MthCalDt']).sum()
    
    if ma_dups > 0:
        print(f"Warning: Found {ma_dups} duplicate (cusip, MthCalDt) pairs in the moving averages dataset")
        # Keep the first occurrence of each duplicate
        df_ma_selected = df_ma_selected.drop_duplicates(subset=['cusip', 'MthCalDt'])
    
    if sentiment_dups > 0:
        print(f"Warning: Found {sentiment_dups} duplicate (cusip, MthCalDt) pairs in the sentiment dataset")
        # Keep the first occurrence of each duplicate
        sentiment_selected = sentiment_selected.drop_duplicates(subset=['cusip', 'MthCalDt'])
    
    # Merge the datasets - using sentiment dataset as the base
    # We're only bringing in the moving average columns from df_ma_selected
    print("Merging datasets with sentiment dataset as the base...")
    
    # First, remove the merge columns from df_ma_selected to avoid duplicates
    ma_only_columns = [col for col in df_ma_selected.columns if col not in ['cusip', 'MthCalDt']]
    
    merged_df = pd.merge(
        sentiment_selected,
        df_ma_selected[['cusip', 'MthCalDt'] + ma_only_columns],
        on=['cusip', 'MthCalDt'],
        how='left'
    )
    
    # Report merge results
    print(f"Merge results:")
    print(f"  Sentiment dataset (base): {sentiment_selected.shape[0]} rows")
    print(f"  Moving averages dataset: {df_ma_selected.shape[0]} rows")
    print(f"  Merged dataset: {merged_df.shape[0]} rows")
    
    # Calculate percentage of rows with moving average data
    ma_coverage = (merged_df['SMA_3'].notna().sum() / merged_df.shape[0]) * 100 if 'SMA_3' in merged_df.columns else 0
    print(f"  Moving average coverage: {ma_coverage:.2f}%")
    
    # Save the merged dataset
    print(f"Saving merged dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print(f"Dataset saved: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    # Basic data quality check
    print("\nData quality check:")
    print("Missing values per column:")
    missing_values = merged_df.isna().sum()
    missing_pct = (missing_values / len(merged_df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    print(missing_df)
    
    # Check unique CUSIPs and dates
    unique_cusips = merged_df['cusip'].nunique()
    unique_dates = merged_df['MthCalDt'].nunique()
    print(f"\nUnique CUSIPs: {unique_cusips}")
    print(f"Unique dates: {unique_dates}")
    print(f"Average observations per CUSIP: {merged_df.shape[0] / unique_cusips:.2f}")
    
    # Check for rows where moving averages were not matched
    ma_missing = merged_df['SMA_3'].isna().sum() if 'SMA_3' in merged_df.columns else merged_df.shape[0]
    print(f"Rows missing moving average data: {ma_missing} ({ma_missing/merged_df.shape[0]*100:.2f}%)")
    
    return merged_df

