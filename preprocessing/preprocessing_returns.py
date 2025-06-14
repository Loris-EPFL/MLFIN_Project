import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


def find_cusips_with_most_unique_data(df, column='CUSIP', date_column='MthCalDt', top_n=10,
                                     min_rows=None, plot=True, figsize=(12, 6),
                                     print_details=True):
    """
    Find and display CUSIPs with the most unique data points after removing duplicates.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str, optional
        Column name containing the identifier (default: 'CUSIP')
    date_column : str, optional
        Column name containing the date (default: 'MthCalDt')
    top_n : int, optional
        Number of top CUSIPs to display
    min_rows : int, optional
        Minimum number of rows required (if specified, only shows CUSIPs with at least this many rows)
    plot : bool, optional
        Whether to plot the results
    figsize : tuple, optional
        Figure size for the plot
    print_details : bool, optional
        Whether to print detailed information about duplicates

    Returns:
    --------
    pandas.DataFrame
        DataFrame with CUSIPs and their unique row counts, sorted by count in descending order
    dict
        Dictionary containing the deduplicated DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    # Count total rows before deduplication
    total_rows = len(df)

    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    # Remove duplicates from the entire dataframe at once
    df_unique = df.drop_duplicates(subset=[column, date_column])

    # Count unique rows per CUSIP
    cusip_counts = df_unique[column].value_counts().reset_index()
    cusip_counts.columns = [column, 'unique_count']

    # Calculate duplicate counts
    total_unique_rows = len(df_unique)
    total_duplicates = total_rows - total_unique_rows

    # Sort by unique count in descending order
    cusip_counts = cusip_counts.sort_values('unique_count', ascending=False).reset_index(drop=True)

    # Filter by minimum rows if specified
    if min_rows is not None:
        cusip_counts = cusip_counts[cusip_counts['unique_count'] >= min_rows]

    # Get top N
    top_cusips = cusip_counts.head(top_n)

    if print_details:
        print(f"Total rows in dataset: {total_rows}")
        print(f"Total unique rows: {total_unique_rows}")
        print(f"Total duplicate rows: {total_duplicates} ({total_duplicates/total_rows*100:.2f}%)")
        print("\nTop CUSIPs with most unique data points:")
        print(top_cusips)

    if plot:
        plt.figure(figsize=figsize)

        # Create horizontal bar chart for unique counts
        bars = plt.barh(top_cusips[column], top_cusips['unique_count'], color='skyblue')

        # Add count labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}',
                    ha='left', va='center')

        plt.xlabel('Number of Unique Data Points')
        plt.ylabel(column)
        plt.title(f'Top {len(top_cusips)} {column}s with Most Unique Data Points')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return cusip_counts, {'df_unique': df_unique}


def add_smoothing_features_to_dataset(df, window_sizes=[3, 6, 12], lowess_frac=0.1,
                                     date_column='MthCalDt', return_column='MthRet',
                                     id_column='CUSIP'):
    """
    Add smoothing features (moving averages and LOWESS) to the entire dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    window_sizes : list, optional
        List of window sizes for moving averages
    lowess_frac : float, optional
        Fraction of data used for LOWESS smoothing (between 0 and 1)
    date_column : str, optional
        Column name containing the date
    return_column : str, optional
        Column name containing the returns
    id_column : str, optional
        Column name containing the identifier (e.g., CUSIP)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added smoothing features
    """

    # Make a copy to avoid modifying the original dataframe
    df_result = df.copy()

    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column])

    # Remove duplicates to ensure clean data
    df_result = df_result.drop_duplicates(subset=[id_column, date_column])

    # Sort by ID and date
    df_result = df_result.sort_values([id_column, date_column])

    # Get unique IDs
    unique_ids = df_result[id_column].unique()

    # Initialize columns for smoothing features
    for window in window_sizes:
        df_result[f'SMA_{window}'] = np.nan
        df_result[f'EMA_{window}'] = np.nan

    df_result['LOWESS'] = np.nan

    # Process each ID
    for id_value in unique_ids:
        print(f"Processing ID: {id_value}")
        # Get mask for this ID
        mask = df_result[id_column] == id_value

        # Get indices for this ID
        indices = df_result.index[mask]

        if len(indices) < 3:  # Skip if too few data points
            continue

        # Calculate Simple Moving Averages
        for window in window_sizes:
            if len(indices) >= window:
                sma_values = df_result.loc[mask, return_column].rolling(window=window).mean()
                df_result.loc[indices, f'SMA_{window}'] = sma_values.values

        # Calculate Exponential Moving Averages
        for window in window_sizes:
            if len(indices) >= window:
                ema_values = df_result.loc[mask, return_column].ewm(span=window, adjust=False).mean()
                df_result.loc[indices, f'EMA_{window}'] = ema_values.values

        # Calculate LOWESS smoothing
        if len(indices) >= 5:  # LOWESS needs at least a few points
            x = np.arange(len(indices))
            y = df_result.loc[indices, return_column].values

            # Apply LOWESS smoothing
            try:
                lowess_result = lowess(y, x, frac=lowess_frac, it=3, return_sorted=False)
                df_result.loc[indices, 'LOWESS'] = lowess_result
            except Exception as e:
                print(f"LOWESS error for {id_value}: {e}")

    return df_result



