
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
    
    def handle_missing_values(self, method='ffill'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        method : str
            Method to handle missing values ('ffill', 'bfill', 'interpolate', or 'drop')
            
        Returns:
        --------
        self : PredictorPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        print("Handling missing values...")
        before_shape = self.data.shape
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0].index.tolist()
        
        if not columns_with_missing:
            print("No missing values found.")
            return self
        
        # Handle missing values based on method
        if method == 'drop':
            # Drop rows with any missing values
            self.data = self.data.dropna()
            
        elif method == 'ffill':
            # Forward fill missing values within each factor group
            if 'name' in self.data.columns:
                self.data = self.data.groupby('name').apply(
                    lambda group: group.fillna(method='ffill')
                ).reset_index(drop=True)
            else:
                self.data = self.data.fillna(method='ffill')
                
            # Handle any remaining missing values at the beginning with backward fill
            self.data = self.data.fillna(method='bfill')
            
        elif method == 'bfill':
            # Backward fill missing values
            if 'name' in self.data.columns:
                self.data = self.data.groupby('name').apply(
                    lambda group: group.fillna(method='bfill')
                ).reset_index(drop=True)
            else:
                self.data = self.data.fillna(method='bfill')
                
            # Handle any remaining missing values at the end with forward fill
            self.data = self.data.fillna(method='ffill')
            
        elif method == 'interpolate':
            # Interpolate missing values within each factor group
            if 'name' in self.data.columns:
                self.data = self.data.groupby('name').apply(
                    lambda group: group.interpolate(method='linear')
                ).reset_index(drop=True)
            else:
                self.data = self.data.interpolate(method='linear')
                
            # Handle any remaining missing values at the edges
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        else:
            raise ValueError("Invalid method. Use 'ffill', 'bfill', 'interpolate', or 'drop'.")
        
        # Report results
        after_shape = self.data.shape
        rows_removed = before_shape[0] - after_shape[0]
        
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing values.")
        
        # Check if any missing values remain
        remaining_missing = self.data.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain.")
        else:
            print("All missing values have been handled.")
        
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
        
        # Pivot the data
        self.factor_data = self.data.pivot(
            index=self.date_column,
            columns='factor_id',
            values=self.return_column
        )
        
        print(f"Pivoted data shape: {self.factor_data.shape}")
        print(f"Created {self.factor_data.shape[1]} factor columns")
        
        return self
    
    def handle_outliers(self, method='winsorize', threshold=3.0):
        """
        Handle outliers in factor returns.
        
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
        
        if method == 'winsorize':
            # Winsorize each factor column (cap at specified quantiles)
            lower_quantile = 0.01
            upper_quantile = 0.99
            
            for col in self.factor_data.columns:
                lower_bound = self.factor_data[col].quantile(lower_quantile)
                upper_bound = self.factor_data[col].quantile(upper_quantile)
                self.factor_data[col] = self.factor_data[col].clip(lower_bound, upper_bound)
                
            print(f"Winsorized factor returns at {lower_quantile:.1%} and {upper_quantile:.1%} quantiles.")
            
        elif method == 'clip':
            # Clip values based on standard deviation
            for col in self.factor_data.columns:
                mean = self.factor_data[col].mean()
                std = self.factor_data[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                self.factor_data[col] = self.factor_data[col].clip(lower_bound, upper_bound)
                
            print(f"Clipped factor returns at ±{threshold} standard deviations.")
            
        elif method == 'zscore':
            # Remove observations with extreme z-scores
            for col in self.factor_data.columns:
                z_scores = np.abs(stats.zscore(self.factor_data[col], nan_policy='omit'))
                self.factor_data[col] = np.where(z_scores > threshold, np.nan, self.factor_data[col])
            
            # Fill the removed values
            self.factor_data = self.factor_data.fillna(method='ffill').fillna(method='bfill')
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
        
        print(f"Normalizing factors using {method} method...")
        
        # Create a copy of the original data for reference
        original_data = self.factor_data.copy()
        
        if window is not None:
            # Time-series normalization with rolling window
            for col in self.factor_data.columns:
                if method == 'standardize':
                    # Z-score normalization with rolling window
                    rolling_mean = self.factor_data[col].rolling(window=window, min_periods=window//2).mean()
                    rolling_std = self.factor_data[col].rolling(window=window, min_periods=window//2).std()
                    self.factor_data[col] = (self.factor_data[col] - rolling_mean) / rolling_std
                    
                elif method == 'minmax':
                    # Min-max scaling with rolling window
                    rolling_min = self.factor_data[col].rolling(window=window, min_periods=window//2).min()
                    rolling_max = self.factor_data[col].rolling(window=window, min_periods=window//2).max()
                    self.factor_data[col] = (self.factor_data[col] - rolling_min) / (rolling_max - rolling_min)
                    
                elif method == 'rank':
                    # Rolling rank transformation
                    self.factor_data[col] = self.factor_data[col].rolling(window=window, min_periods=window//2).apply(
                        lambda x: (pd.Series(x).rank() - 1) / (len(x) - 1)
                    )
            
            print(f"Applied rolling {method} normalization with window size {window}.")
            
        else:
            # Cross-sectional normalization
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
                normalized_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
                self.factor_data = normalized_data
                
            elif method == 'rank':
                # Rank transformation (0 to 1)
                self.factor_data = original_data.rank(pct=True)
                
            else:
                raise ValueError("Invalid method. Use 'standardize', 'robust', 'minmax', or 'rank'.")
            
            print(f"Applied {method} normalization across all factors.")
        
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
        Plot factor returns over time.
        
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
        
        if factors is None:
            # Select a sample of factors
            factors = self.factor_data.columns[:n_factors]
        
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
        Plot correlation matrix of factors.
        
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
        
        # Select a subset of factors if there are too many
        if self.factor_data.shape[1] > n_factors:
            # Select factors with highest variance
            variances = self.factor_data.var().sort_values(ascending=False)
            selected_factors = variances.index[:n_factors]
            corr_data = self.factor_data[selected_factors].corr()
        else:
            corr_data = self.factor_data.corr()
        
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
    
    def get_data(self):
        """
        Get the processed factor data.
        
        Returns:
        --------
        pandas.DataFrame
            The processed factor data
        """
        return self.factor_data.copy() if self.factor_data is not None else None
    
    def export(self, file_path, format='parquet'):
        """
        Export the processed factor data.
        
        Parameters:
        -----------
        file_path : str
            Path to save the processed data
        format : str
            Format to save the data ('parquet' or 'csv')
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if self.factor_data is None:
            raise ValueError("No processed factor data to export.")
        
        # Create directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == 'parquet':
            if not file_path.endswith('.parquet'):
                file_path += '.parquet'
            self.factor_data.to_parquet(file_path)
        elif format.lower() == 'csv':
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.factor_data.to_csv(file_path)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")
        
        print(f"Factor data exported to {file_path}")
        print(f"Final shape: {self.factor_data.shape}")
        
        return file_path
