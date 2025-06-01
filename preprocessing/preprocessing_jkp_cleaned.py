
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats

class JkpPreprocessor:
    """
    A class for preprocessing financial predictor datasets, particularly factor-based predictors.
    
    This class provides methods to load, clean, transform, and prepare predictor data
    for machine learning applications in finance.
    """
    
    def __init__(self, date_column='date', return_column='ret'):
        """
        Initialize the PredictorPreprocessor with column names.
        
        Parameters:
        -----------
        date_column : str
            Column name for the date
        return_column : str
            Column name for the factor returns
        """
        self.date_column = date_column
        self.return_column = return_column
        self.data = None
        self.factor_data = None  # Will hold pivoted factor data
        self.original_shape = None
        self.scaler = None
        self.pca = None
        
    def load(self, file_path, nrows=None):
        """
        Load the predictor dataset from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        nrows : int, optional
            Number of rows to load (useful for large datasets)
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, nrows=nrows)
            elif file_path.endswith('.parquet'):
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
                
            self.original_shape = self.data.shape
            print(f"Loaded dataset with shape: {self.original_shape}")
            return self
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def explore_data(self):
        """
        Explore the dataset to understand its structure and identify issues.
        
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Basic info
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.data.shape}")
        print("\n=== Data Types ===")
        print(self.data.dtypes)
        
        # Missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).sort_values('Missing Values', ascending=False)
        print("\n=== Missing Values ===")
        print(missing_df[missing_df['Missing Values'] > 0])
        
        # Factor statistics
        print("\n=== Factor Return Statistics ===")
        print(self.data[self.return_column].describe())
        
        # Count unique values for categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n=== Categorical Columns ===")
            for col in categorical_cols:
                unique_values = self.data[col].nunique()
                print(f"{col}: {unique_values} unique values")
                if unique_values < 20:  # Show all values if there aren't too many
                    print(f"Values: {sorted(self.data[col].unique())}")
        
        # Time range
        if pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            date_range = self.data[self.date_column]
        else:
            date_range = pd.to_datetime(self.data[self.date_column])
            
        print("\n=== Time Range ===")
        print(f"From {date_range.min()} to {date_range.max()}")
        
        # Factor count
        factor_count = self.data['name'].nunique() if 'name' in self.data.columns else 0
        print(f"\n=== Factor Count ===")
        print(f"Dataset contains {factor_count} unique factors")
        
        return {
            'shape': self.data.shape,
            'missing': missing_df,
            'factor_stats': self.data[self.return_column].describe() if self.return_column in self.data.columns else None,
            'factor_count': factor_count
        }
    
    def explore_factor_names(self):
        """
        Explore all unique factor names in the dataset and provide information about them.
        
        This function prints all unique values in the 'name' column, along with their
        frequency and a sample of return values to help understand what each factor represents.
        
        Returns:
        --------
        self : JkpPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Get all unique factor names
        factor_names = sorted(self.data['name'].unique())
        
        print(f"Found {len(factor_names)} unique factors in the dataset:")
        print("\n=== FACTOR SUMMARY ===")
        
        # Create a summary table
        summary_data = []
        
        for factor in factor_names:
            # Get data for this factor
            factor_data = self.data[self.data['name'] == factor]
            
            # Calculate statistics
            count = len(factor_data)
            mean_return = factor_data['ret'].mean()
            std_return = factor_data['ret'].std()
            min_return = factor_data['ret'].min()
            max_return = factor_data['ret'].max()
            
            # Get date range
            if not pd.api.types.is_datetime64_dtype(factor_data[self.date_column]):
                factor_data[self.date_column] = pd.to_datetime(factor_data[self.date_column])
            
            start_date = factor_data[self.date_column].min()
            end_date = factor_data[self.date_column].max()
            
            # Add to summary data
            summary_data.append({
                'Factor': factor,
                'Count': count,
                'Mean Return': mean_return,
                'Std Dev': std_return,
                'Min': min_return,
                'Max': max_return,
                'Start Date': start_date,
                'End Date': end_date
            })
        
        # Convert to DataFrame for nice display
        summary_df = pd.DataFrame(summary_data)
        
        # Group factors by potential categories
        sentiment_related = []
        momentum_related = []
        volatility_related = []
        valuation_related = []
        quality_related = []
        growth_related = []
        other_factors = []
        
        # Simple categorization based on factor name patterns
        for factor in factor_names:
            factor_lower = factor.lower()
            
            # Sentiment and market sentiment indicators
            if any(term in factor_lower for term in ['sentiment', 'ret_1_0', 'rskew', 'iskew', 'niq_su', 'saleq_su']):
                sentiment_related.append(factor)
            
            # Momentum factors
            elif any(term in factor_lower for term in ['ret_', 'momentum', 'resff3']):
                momentum_related.append(factor)
            
            # Volatility factors
            elif any(term in factor_lower for term in ['vol', 'beta', 'ivol', 'rvol', 'bidask']):
                volatility_related.append(factor)
            
            # Valuation factors
            elif any(term in factor_lower for term in ['me', 'bev', 'ebitda', 'fcf', 'debt', 'sale_me']):
                valuation_related.append(factor)
            
            # Quality factors
            elif any(term in factor_lower for term in ['score', 'qmj', 'safety', 'prof', 'mispricing']):
                quality_related.append(factor)
            
            # Growth factors
            elif any(term in factor_lower for term in ['gr1', 'gr2', 'gr3', 'growth']):
                growth_related.append(factor)
            
            # Other factors
            else:
                other_factors.append(factor)
        
        # Print categorized factors
        print("\n=== FACTORS BY CATEGORY ===")
        
        print("\nSentiment-Related Factors:")
        for factor in sorted(sentiment_related):
            print(f"  - {factor}")
        
        print("\nMomentum Factors:")
        for factor in sorted(momentum_related):
            print(f"  - {factor}")
        
        print("\nVolatility Factors:")
        for factor in sorted(volatility_related):
            print(f"  - {factor}")
        
        print("\nValuation Factors:")
        for factor in sorted(valuation_related):
            print(f"  - {factor}")
        
        print("\nQuality Factors:")
        for factor in sorted(quality_related):
            print(f"  - {factor}")
        
        print("\nGrowth Factors:")
        for factor in sorted(growth_related):
            print(f"  - {factor}")
        
        print("\nOther Factors:")
        for factor in sorted(other_factors):
            print(f"  - {factor}")
        
        # Print summary statistics
        print("\n=== FACTOR STATISTICS ===")
        pd.set_option('display.max_rows', None)
        print(summary_df.sort_values('Factor')[['Factor', 'Count', 'Mean Return', 'Std Dev', 'Min', 'Max']])
        pd.reset_option('display.max_rows')
        
        return self

    
    
    def convert_dates(self):
        """
        Convert date column to datetime format.
        
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        # Sort by date
        self.data = self.data.sort_values(self.date_column)
        
        print("Date conversion complete.")
        return self
    
    def handle_missing_values(self, data_type='raw', method='ffill'):
        """
        Handle missing values in any of the dataset types.
        
        Parameters:
        -----------
        data_type : str
            Type of data to process ('raw', 'pivoted', or 'factor_columns')
        method : str
            Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop', or 'drop_sparse')
            
        Returns:
        --------
        self : JkpPreprocessor
            Returns self for method chaining
        """
        # Determine which dataset to process
        if data_type == 'raw':
            if self.data is None:
                raise ValueError("No raw data loaded. Call load() first.")
            data = self.data
            data_name = "raw data"
        elif data_type == 'pivoted':
            if self.factor_data is None:
                raise ValueError("No pivoted factor data. Call pivot_factors() first.")
            data = self.factor_data
            data_name = "pivoted factor data"
        elif data_type == 'factor_columns':
            if not hasattr(self, 'factor_columns_data') or self.factor_columns_data is None:
                raise ValueError("No factor columns data. Call create_factor_columns_dataset() first.")
            data = self.factor_columns_data
            data_name = "factor columns data"
        else:
            raise ValueError("Invalid data_type. Use 'raw', 'pivoted', or 'factor_columns'.")
        
        print(f"\nHandling missing values in {data_name}...")
        
        # Check for missing values
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        print(f"Missing values: {missing_cells} out of {total_cells} cells ({missing_pct:.2f}%)")
        
        if missing_cells == 0:
            print("No missing values found.")
            return self
        
        # For pivoted and factor_columns data, check for sparse columns
        if data_type in ['pivoted', 'factor_columns']:
            missing_by_col = data.isna().mean().sort_values(ascending=False)
            sparse_columns = missing_by_col[missing_by_col > 0.5].index.tolist()
            
            if sparse_columns:
                print(f"Found {len(sparse_columns)} sparse columns with >50% missing values")
        
        # Store original shape for reporting
        before_shape = data.shape
        
        # Handle missing values based on method
        if method == 'drop':
            # Drop rows with any missing values
            data = data.dropna()
            
        elif method == 'drop_sparse' and data_type in ['pivoted', 'factor_columns']:
            # Drop columns with too many missing values, then drop rows with any remaining missing values
            if sparse_columns:
                print(f"Dropping {len(sparse_columns)} sparse columns")
                data = data.drop(columns=sparse_columns)
            
            data = data.dropna()
            
        elif method == 'ffill':
            # Forward fill missing values
            if data_type == 'raw' and 'name' in data.columns:
                # For raw data, fill within each factor group
                data = data.groupby('name').apply(
                    lambda group: group.fillna(method='ffill')
                ).reset_index(drop=True)
            else:
                data = data.fillna(method='ffill')
                
            # Handle any remaining missing values at the beginning with backward fill
            data = data.fillna(method='bfill')
            
        elif method == 'bfill':
            # Backward fill missing values
            if data_type == 'raw' and 'name' in data.columns:
                # For raw data, fill within each factor group
                data = data.groupby('name').apply(
                    lambda group: group.fillna(method='bfill')
                ).reset_index(drop=True)
            else:
                data = data.fillna(method='bfill')
                
            # Handle any remaining missing values at the end with forward fill
            data = data.fillna(method='ffill')
            
        elif method == 'interpolate':
            # Interpolate missing values
            if data_type == 'raw' and 'name' in data.columns:
                # For raw data, interpolate within each factor group
                data = data.groupby('name').apply(
                    lambda group: group.interpolate(method='linear')
                ).reset_index(drop=True)
            else:
                data = data.interpolate(method='linear', axis=0)
                
            # Handle any remaining missing values at the edges
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        else:
            valid_methods = "'ffill', 'bfill', 'interpolate', 'drop'"
            if data_type in ['pivoted', 'factor_columns']:
                valid_methods += ", or 'drop_sparse'"
            raise ValueError(f"Invalid method. Use {valid_methods}.")
        
        # Report results
        after_shape = data.shape
        
        if data_type == 'raw':
            rows_removed = before_shape[0] - after_shape[0]
            if rows_removed > 0:
                print(f"Removed {rows_removed} rows with missing values.")
        else:
            rows_removed = before_shape[0] - after_shape[0]
            cols_removed = before_shape[1] - after_shape[1]
            if rows_removed > 0:
                print(f"Removed {rows_removed} rows with missing values.")
            if cols_removed > 0:
                print(f"Removed {cols_removed} columns with missing values.")
        
        # Check if any missing values remain
        remaining_missing = data.isna().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain after processing")
        else:
            print("All missing values have been handled")
        
        # Update the appropriate dataset
        if data_type == 'raw':
            self.data = data
        elif data_type == 'pivoted':
            self.factor_data = data
            print("\n=== Processed Factor Data ===")
            print(self.factor_data.head())
        elif data_type == 'factor_columns':
            self.factor_columns_data = data
            print("\n=== Factor Columns Data ===")
            print(self.factor_columns_data.head())
        
        return self

    def pivot_factors(self):
        """
        Pivot the factor data to create a time series of factor returns.
        Each row will represent a date, and each column will be a factor.
        
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        if 'name' not in self.data.columns:
            raise ValueError("Data must contain a 'name' column to identify factors.")
        
        print("Pivoting factor data...")
        
        # Create a unique factor identifier if needed
        if 'location' in self.data.columns and 'weighting' in self.data.columns:
            self.data['factor_id'] = self.data['name'] + '_' + self.data['location'] + '_' + self.data['weighting']
        else:
            self.data['factor_id'] = self.data['name']
        
        # Pivot the returns data
        self.factor_data = self.data.pivot(
            index=self.date_column,
            columns='factor_id',
            values=self.return_column
        )
        
        # If n_stocks column exists, pivot it as well
        if 'n_stocks' in self.data.columns:
            n_stocks_pivot = self.data.pivot(
                index=self.date_column,
                columns='factor_id',
                values='n_stocks'
            )
            
            # Rename columns to indicate they represent n_stocks
            n_stocks_pivot.columns = [f"{col}_n_stocks" for col in n_stocks_pivot.columns]
            
            # Join with the factor data
            self.factor_data = pd.concat([self.factor_data, n_stocks_pivot], axis=1)
            
            print(f"Added {len(n_stocks_pivot.columns)} n_stocks columns to processed data")
        
        print(f"Pivoted data shape: {self.factor_data.shape}")
        print(f"Created {self.factor_data.shape[1]} columns")
        
        return self

    
    
   
    def handle_outliers(self, method='winsorize', threshold=3.0):
        """
        Handle outliers in factor returns (excluding n_stocks columns).
        
        Parameters:
        -----------
        method : str
            Method to handle outliers ('winsorize', 'clip', or 'zscore')
        threshold : float
            Threshold for outlier detection (standard deviations for zscore,
            or quantile limits for winsorize)
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        print(f"Handling outliers using {method} method...")
        
        # Filter out n_stocks columns
        factor_cols = [col for col in self.factor_data.columns if not col.endswith('_n_stocks')]
        n_stocks_cols = [col for col in self.factor_data.columns if col.endswith('_n_stocks')]
        
        print(f"Processing {len(factor_cols)} factor columns (excluding {len(n_stocks_cols)} n_stocks columns)")
        
        if method == 'winsorize':
            # Winsorize each factor column (cap at specified quantiles)
            lower_quantile = 0.01
            upper_quantile = 0.99
            
            for col in factor_cols:
                lower_bound = self.factor_data[col].quantile(lower_quantile)
                upper_bound = self.factor_data[col].quantile(upper_quantile)
                self.factor_data[col] = self.factor_data[col].clip(lower_bound, upper_bound)
                
            print(f"Winsorized factor returns at {lower_quantile:.1%} and {upper_quantile:.1%} quantiles.")
            
        elif method == 'clip':
            # Clip values based on standard deviation
            for col in factor_cols:
                mean = self.factor_data[col].mean()
                std = self.factor_data[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                self.factor_data[col] = self.factor_data[col].clip(lower_bound, upper_bound)
                
            print(f"Clipped factor returns at ±{threshold} standard deviations.")
            
        elif method == 'zscore':
            # Remove observations with extreme z-scores
            for col in factor_cols:
                z_scores = np.abs(stats.zscore(self.factor_data[col], nan_policy='omit'))
                self.factor_data[col] = np.where(z_scores > threshold, np.nan, self.factor_data[col])
            
            # Fill the removed values
            for col in factor_cols:
                self.factor_data[col] = self.factor_data[col].fillna(method='ffill').fillna(method='bfill')
            
            print(f"Replaced outliers with z-score > {threshold} with interpolated values.")
            
        else:
            raise ValueError("Invalid method. Use 'winsorize', 'clip', or 'zscore'.")
        
        return self

   
    def normalize_factors(self, method='standardize', window=None):
        """
        Normalize factor returns using various methods.
        
        Parameters:
        -----------
        method : str
            Normalization method ('standardize', 'robust', 'minmax', or 'rank')
        window : int, optional
            Rolling window size for time-series normalization
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        # Check if factor_data is empty or contains only NaN values
        if self.factor_data.empty:
            print("Warning: Factor data is empty. Skipping normalization.")
            return self
        
        # Check for columns with all NaN values
        nan_columns = self.factor_data.columns[self.factor_data.isna().all()].tolist()
        if nan_columns:
            print(f"Warning: Removing {len(nan_columns)} columns with all NaN values before normalization.")
            self.factor_data = self.factor_data.drop(columns=nan_columns)
        
        # Check if any data remains after removing NaN columns
        if self.factor_data.empty:
            print("Warning: No valid data remains after removing NaN columns. Skipping normalization.")
            return self
        
        print(f"Normalizing factors using {method} method...")
        
        # Create a copy of the original data for reference
        original_data = self.factor_data.copy()
        
        # Fill any remaining NaN values before normalization
        original_data = original_data.fillna(method='ffill').fillna(method='bfill')
        
        if window is not None:
            # Time-series normalization with rolling window
            for col in self.factor_data.columns:
                if method == 'standardize':
                    # Z-score normalization with rolling window
                    rolling_mean = self.factor_data[col].rolling(window=window, min_periods=window//2).mean()
                    rolling_std = self.factor_data[col].rolling(window=window, min_periods=window//2).std()
                    # Handle zero standard deviation
                    rolling_std = rolling_std.replace(0, 1)
                    self.factor_data[col] = (self.factor_data[col] - rolling_mean) / rolling_std
                    
                elif method == 'minmax':
                    # Min-max scaling with rolling window
                    rolling_min = self.factor_data[col].rolling(window=window, min_periods=window//2).min()
                    rolling_max = self.factor_data[col].rolling(window=window, min_periods=window//2).max()
                    # Handle zero range
                    range_values = rolling_max - rolling_min
                    range_values = range_values.replace(0, 1)
                    self.factor_data[col] = (self.factor_data[col] - rolling_min) / range_values
                    
                elif method == 'rank':
                    # Rolling rank transformation
                    self.factor_data[col] = self.factor_data[col].rolling(window=window, min_periods=window//2).apply(
                        lambda x: (pd.Series(x).rank() - 1) / (len(x) - 1) if len(x) > 1 else 0.5
                    )
            
            print(f"Applied rolling {method} normalization with window size {window}.")
            
        else:
            # Cross-sectional normalization
            try:
                if method == 'standardize':
                    # Z-score standardization
                    self.scaler = StandardScaler()
                    normalized_data = self.scaler.fit_transform(original_data)
                    self.factor_data = pd.DataFrame(
                        normalized_data, 
                        index=original_data.index, 
                        columns=original_data.columns
                    )
                    
                elif method == 'robust':
                    # Robust scaling (less sensitive to outliers)
                    self.scaler = RobustScaler()
                    normalized_data = self.scaler.fit_transform(original_data)
                    self.factor_data = pd.DataFrame(
                        normalized_data, 
                        index=original_data.index, 
                        columns=original_data.columns
                    )
                    
                elif method == 'minmax':
                    # Min-max scaling to [0, 1]
                    min_vals = original_data.min()
                    max_vals = original_data.max()
                    # Handle zero range
                    range_vals = max_vals - min_vals
                    range_vals = range_vals.replace(0, 1)
                    self.factor_data = (original_data - min_vals) / range_vals
                    
                elif method == 'rank':
                    # Rank transformation (0 to 1)
                    self.factor_data = original_data.rank(pct=True)
                    
                else:
                    raise ValueError("Invalid method. Use 'standardize', 'robust', 'minmax', or 'rank'.")
                
                print(f"Applied {method} normalization across all factors.")
            
            except Exception as e:
                print(f"Error during normalization: {e}")
                print("Skipping normalization step.")
                self.factor_data = original_data
        
        # Handle any NaNs created during normalization
        self.factor_data = self.factor_data.fillna(method='ffill').fillna(method='bfill')
        
        return self


    def create_lagged_features(self, lags=[1, 3, 6, 12]):
        """
        Create lagged versions of factor returns as additional features.
        
        Parameters:
        -----------
        lags : list
            List of lag periods to create
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        print(f"Creating lagged features with lags: {lags}...")
        
        # Store original columns
        original_columns = self.factor_data.columns.tolist()
        
        # Create lagged features
        for lag in lags:
            for col in original_columns:
                lag_col_name = f"{col}_lag{lag}"
                self.factor_data[lag_col_name] = self.factor_data[col].shift(lag)
        
        # Drop rows with NaN values created by lagging
        max_lag = max(lags)
        self.factor_data = self.factor_data.iloc[max_lag:].copy()
        
        print(f"Created {len(lags) * len(original_columns)} lagged features.")
        print(f"First {max_lag} rows dropped due to lagging.")
        
        return self

    def create_rolling_features(self, windows=[3, 6, 12], functions=['mean', 'std']):
        """
        Create rolling window statistics as additional features.
        
        Parameters:
        -----------
        windows : list
            List of window sizes for rolling calculations
        functions : list
            List of functions to apply ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        print(f"Creating rolling features with windows: {windows}...")
        
        # Store original columns
        original_columns = self.factor_data.columns.tolist()
        
        # Map function names to actual functions
        func_map = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum
        }
        
        # Create rolling features
        for window in windows:
            for func_name in functions:
                if func_name not in func_map:
                    print(f"Warning: Function '{func_name}' not recognized. Skipping.")
                    continue
                    
                func = func_map[func_name]
                
                for col in original_columns:
                    feature_name = f"{col}_{func_name}{window}"
                    self.factor_data[feature_name] = self.factor_data[col].rolling(
                        window=window, min_periods=window//2
                    ).apply(func)
        
        # Fill any NaN values created by rolling windows
        self.factor_data = self.factor_data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Created {len(windows) * len(functions) * len(original_columns)} rolling features.")
        
        return self

    def reduce_dimensions(self, n_components=None, variance_threshold=0.95):
        """
        Reduce dimensionality of the factor data using PCA.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep
        variance_threshold : float
            Minimum cumulative explained variance ratio
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        # Determine number of components
        if n_components is None:
            # Start with a large number of components
            temp_pca = PCA(n_components=min(self.factor_data.shape[0], self.factor_data.shape[1]))
            temp_pca.fit(self.factor_data)
            
            # Find number of components needed to reach variance threshold
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"Reducing dimensions using PCA with {n_components} components...")
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(self.factor_data)
        
        # Create new dataframe with PCA results
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        self.factor_data = pd.DataFrame(
            pca_result,
            index=self.factor_data.index,
            columns=pca_columns
        )
        
        # Report explained variance
        explained_variance = sum(self.pca.explained_variance_ratio_)
        print(f"PCA completed. {n_components} components explain {explained_variance:.2%} of variance.")
        
        return self

    def plot_factor_returns(self, factors=None, n_factors=5):
        """
        Plot factor returns over time (excluding n_stocks columns).
        
        Parameters:
        -----------
        factors : list, optional
            List of factor names to plot
        n_factors : int
            Number of factors to plot if factors not specified
            
        Returns:
        --------
        None
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        # Filter out n_stocks columns
        factor_cols = [col for col in self.factor_data.columns if not col.endswith('_n_stocks')]
        
        if factors is None:
            # Select a sample of factors
            factors = factor_cols[:n_factors]
        else:
            # Filter out any n_stocks columns from the provided list
            factors = [f for f in factors if not f.endswith('_n_stocks')]
            
            if not factors:
                print("Warning: All specified factors were n_stocks columns. Selecting first few regular factors instead.")
                factors = factor_cols[:n_factors]
        
        plt.figure(figsize=(12, 8))
        
        for factor in factors:
            if factor in self.factor_data.columns:
                plt.plot(self.factor_data.index, self.factor_data[factor], label=factor)
        
        plt.title('Factor Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_factor_correlations(self, n_factors=20):
        """
        Plot correlation matrix of factors (excluding n_stocks columns).
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to include in correlation plot
            
        Returns:
        --------
        None
        """
        if self.factor_data is None:
            raise ValueError("No pivoted factor data. Call pivot_factors() first.")
        
        # Filter out n_stocks columns
        factor_cols = [col for col in self.factor_data.columns if not col.endswith('_n_stocks')]
        
        print(f"Calculating correlations for {len(factor_cols)} factor columns (excluding n_stocks columns)")
        
        # Select a subset of factors if there are too many
        if len(factor_cols) > n_factors:
            # Select factors with highest variance
            variances = self.factor_data[factor_cols].var().sort_values(ascending=False)
            selected_factors = variances.index[:n_factors]
            corr_data = self.factor_data[selected_factors].corr()
        else:
            corr_data = self.factor_data[factor_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        sns.heatmap(
            corr_data,
            mask=mask,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            annot=True if corr_data.shape[0] <= 15 else False,
            fmt='.2f' if corr_data.shape[0] <= 15 else '',
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'}
        )
        
        plt.title('Factor Correlation Matrix')
        plt.tight_layout()
        plt.show()


    def get_data(self, data_type='processed'):
        """
        Get any of the dataset types.
        
        Parameters:
        -----------
        data_type : str
            Type of data to retrieve ('raw', 'processed', or 'factor_columns')
            
        Returns:
        --------
        pandas.DataFrame
            The requested dataset
        """
        if data_type == 'raw':
            if self.data is None:
                raise ValueError("No raw data loaded. Call load() first.")
            return self.data.copy()
        
        elif data_type == 'processed':
            if self.factor_data is None:
                raise ValueError("No processed factor data available. Call pivot_factors() first.")
            return self.factor_data.copy()
        
        elif data_type == 'factor_columns':
            if not hasattr(self, 'factor_columns_data') or self.factor_columns_data is None:
                raise ValueError("No factor columns data. Call create_factor_columns_dataset() first.")
            return self.factor_columns_data.copy()
        
        else:
            raise ValueError("Invalid data_type. Use 'raw', 'processed', or 'factor_columns'.")

    def export_data(self, file_path, data_type='processed', format='parquet', date_column_name='date'):
        """
        Export any of the dataset types to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the data
        data_type : str
            Type of data to export ('raw', 'processed', or 'factor_columns')
        format : str
            Format to save the data ('parquet' or 'csv')
        date_column_name : str
            Name to use for the date index column when exporting to CSV
            
        Returns:
        --------
        str
            Path to the saved file
        """
        # Determine which dataset to export
        if data_type == 'raw':
            if self.data is None:
                raise ValueError("No raw data loaded. Call load() first.")
            data = self.data
            data_name = "raw data"
            has_date_index = False
        elif data_type == 'processed':
            if self.factor_data is None:
                raise ValueError("No processed factor data available. Call pivot_factors() first.")
            data = self.factor_data
            data_name = "processed factor data"
            has_date_index = True
        elif data_type == 'factor_columns':
            if not hasattr(self, 'factor_columns_data') or self.factor_columns_data is None:
                raise ValueError("No factor columns data. Call create_factor_columns_dataset() first.")
            data = self.factor_columns_data
            data_name = "factor columns data"
            has_date_index = True
        else:
            raise ValueError("Invalid data_type. Use 'raw', 'processed', or 'factor_columns'.")
        
        # Create directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == 'parquet':
            if not file_path.endswith('.parquet'):
                file_path += '.parquet'
            data.to_parquet(file_path)
        elif format.lower() == 'csv':
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            
            # Handle date index for processed and factor_columns data
            if has_date_index:
                # Reset index to make the date a column with the specified name
                df_to_export = data.reset_index()
                df_to_export.rename(columns={'index': date_column_name}, inplace=True)
                df_to_export.to_csv(file_path, index=False)
            else:
                # Raw data doesn't have a date index
                data.to_csv(file_path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")
        
        print(f"{data_name.capitalize()} exported to {file_path}")
        print(f"Shape: {data.shape}")
        
        return file_path

    def select_factors(self, factors_to_keep):
        """
        Select specific JKP factors to keep in the dataset.
        
        Parameters:
        -----------
        factors_to_keep : list
            List of factor names to keep (e.g., ['age', 'ret_12_1', 'rvol_21d'])
                
        Returns:
        --------
        self : JkpPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Check if any of the requested factors exist in the dataset
        available_factors = self.data['name'].unique()
        found_factors = [f for f in factors_to_keep if f in available_factors]
        missing_factors = [f for f in factors_to_keep if f not in available_factors]
        
        if not found_factors:
            raise ValueError("None of the requested factors were found in the dataset.")
        
        if missing_factors:
            print(f"Warning: The following factors were not found in the dataset: {missing_factors}")
        
        # Filter to keep only specified factors
        original_rows = len(self.data)
        original_factors = len(self.data['name'].unique())
        
        self.data = self.data[self.data['name'].isin(found_factors)]
        
        # Report results
        final_rows = len(self.data)
        final_factors = len(self.data['name'].unique())
        
        print(f"Selected {final_factors} out of {original_factors} factors")
        print(f"Rows before: {original_rows}, Rows after: {final_rows}")
        print("\nKept factors:")
        for factor in sorted(self.data['name'].unique()):
            print(f"  - {factor}")
        
        return self

    def filter_by_date(self, start_date=None, end_date=None):
        """
        Filter the dataset to include only data within a specified date range.
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            The earliest date to include (format: 'YYYY-MM-DD')
        end_date : str or datetime, optional
            The latest date to include (format: 'YYYY-MM-DD')
                
        Returns:
        --------
        self : JkpPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Convert dates if needed
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        original_rows = len(self.data)
        original_date_range = f"{self.data[self.date_column].min():%Y-%m-%d} to {self.data[self.date_column].max():%Y-%m-%d}"
        
        # Apply filters
        if start_date:
            start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data[self.date_column] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            self.data = self.data[self.data[self.date_column] <= end_date]
        
        # Report results
        final_rows = len(self.data)
        final_date_range = f"{self.data[self.date_column].min():%Y-%m-%d} to {self.data[self.date_column].max():%Y-%m-%d}"
        
        print(f"Original date range: {original_date_range}")
        print(f"Filtered date range: {final_date_range}")
        print(f"Rows before: {original_rows}, Rows after: {final_rows}")
        
        return self

    def create_factor_columns_dataset(self):
        """
        Create a dataset where each factor from the 'name' column becomes its own column.
        
        This method creates a DataFrame where:
        - Each row represents a date
        - Each column represents a factor (using original factor names from the 'name' column)
        - Values are the factor returns for that date
        
        Returns:
        --------
        self : JkpPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        if 'name' not in self.data.columns:
            raise ValueError("Data must contain a 'name' column to identify factors.")
        
        print("Creating dataset with factor columns...")
        
        # Get unique dates and factor names
        unique_dates = sorted(self.data[self.date_column].unique())
        unique_factors = sorted(self.data['name'].unique())
        
        print(f"Found {len(unique_factors)} unique factors across {len(unique_dates)} dates")
        
        # Create an empty DataFrame with dates as index
        factor_columns_data = pd.DataFrame(index=pd.DatetimeIndex(unique_dates))
        
        # For each factor, extract its time series and add as a column
        for factor in unique_factors:
            # Get data for this factor
            factor_data = self.data[self.data['name'] == factor]
            
            # Create a Series with the factor returns indexed by date
            factor_series = pd.Series(
                factor_data[self.return_column].values,
                index=pd.DatetimeIndex(factor_data[self.date_column]),
                name=factor
            )
            
            # Add the factor as a column to the DataFrame
            factor_columns_data[factor] = factor_series
            
            # If n_stocks exists, add it as a column too
            if 'n_stocks' in self.data.columns:
                n_stocks_series = pd.Series(
                    factor_data['n_stocks'].values,
                    index=pd.DatetimeIndex(factor_data[self.date_column]),
                    name=f"{factor}_n_stocks"
                )
                factor_columns_data[f"{factor}_n_stocks"] = n_stocks_series
        
        # Store the factor columns data
        self.factor_columns_data = factor_columns_data
        
        print(f"Created factor columns dataset with shape: {self.factor_columns_data.shape}")
        print(f"Contains {len(unique_factors)} factor columns")
        if 'n_stocks' in self.data.columns:
            print(f"Also contains {len(unique_factors)} n_stocks columns")
        
        return self

    
    def get_factor_columns_data(self):
        """
        Get the factor columns dataset.
        
        Returns:
        --------
        pandas.DataFrame
            The factor columns dataset
        """
        if not hasattr(self, 'factor_columns_data') or self.factor_columns_data is None:
            raise ValueError("No factor columns data. Call create_factor_columns_dataset() first.")
        
        return self.factor_columns_data.copy()

   