import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ReturnPreprocessor:
    """
    A lightweight class for loading and basic preprocessing of financial returns datasets.
    
    This class provides methods to load the data, check for missing values,
    and prepare it for machine learning applications in finance.
    """
    
    def __init__(self, id_column='PERMNO', date_column='MthCalDt', 
                 return_column='MthRet', market_return_column='sprtrn'):
        """
        Initialize the ReturnPreprocessor with column names.
        
        Parameters:
        -----------
        id_column : str
            Column name for the security identifier
        date_column : str
            Column name for the date
        return_column : str
            Column name for the return values
        market_return_column : str
            Column name for the market return (benchmark)
        """
        self.id_column = id_column
        self.date_column = date_column
        self.return_column = return_column
        self.market_return_column = market_return_column
        self.data = None
        
    def load(self, file_path, nrows=None):
        """
        Load the returns dataset from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        nrows : int, optional
            Number of rows to load (useful for large datasets)
            
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, nrows=nrows)
            elif file_path.endswith('.parquet'):
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
                
            print(f"Loaded dataset with shape: {self.data.shape}")
            return self
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame showing missing value counts and percentages
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Count missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).sort_values('Missing Values', ascending=False)
        
        # Print summary of missing values
        print("\n=== Missing Values Summary ===")
        print(missing_df[missing_df['Missing Values'] > 0])
        
        return missing_df
    
    def convert_dates(self):
        """
        Convert date column to datetime format.
        
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        # Sort by ID and date
        self.data = self.data.sort_values([self.id_column, self.date_column])
        
        print("Date conversion complete.")
        return self
    
    def summarize(self):
        """
        Provide a summary of the dataset.
        
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
        
        # Return statistics
        print("\n=== Return Statistics ===")
        print(self.data[self.return_column].describe())
        
        # Count unique values for categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n=== Categorical Columns ===")
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                print(f"{col}: {unique_count} unique values")
        
        return {
            'shape': self.data.shape,
            'return_stats': self.data[self.return_column].describe()
        }
    
    def plot_returns_overview(self, sample_size=5):
        """
        Create basic plots to visualize the return data.
        
        Parameters:
        -----------
        sample_size : int
            Number of securities to sample for time series plot
            
        Returns:
        --------
        None
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Return distribution
        sns.histplot(self.data[self.return_column].dropna(), kde=True, ax=axes[0])
        axes[0].set_title('Return Distribution')
        axes[0].set_xlabel('Return')
        
        # Return over time for a sample of securities
        sample_ids = self.data[self.id_column].unique()[:sample_size]
        sample_data = self.data[self.data[self.id_column].isin(sample_ids)]
        
        for id_val in sample_ids:
            id_data = sample_data[sample_data[self.id_column] == id_val]
            axes[1].plot(id_data[self.date_column], id_data[self.return_column], label=str(id_val))
        
        axes[1].set_title('Returns Over Time (Sample)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Return')
        axes[1].legend(title=self.id_column)
        
        plt.tight_layout()
        plt.show()
    
    def get_data(self):
        """
        Get the processed dataframe.
        
        Returns:
        --------
        pandas.DataFrame
            The processed dataframe
        """
        return self.data.copy() if self.data is not None else None
    
    def export(self, file_path, format='parquet'):
        """
        Export the dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to save the data
        format : str
            Format to save the data ('parquet' or 'csv')
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if self.data is None:
            raise ValueError("No data to export. Load data first.")
        
        # Create directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == 'parquet':
            if not file_path.endswith('.parquet'):
                file_path += '.parquet'
            self.data.to_parquet(file_path)
        elif format.lower() == 'csv':
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.data.to_csv(file_path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")
        
        print(f"Data exported to {file_path}")
        
        return file_path
    
    def select_essential_columns(self):
        """
        Select only PERMNO, PERMCO, MthCalDt, MthRet, and sprtrn columns.
        
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        essential_columns = [
            'PERMNO',      # Unique security identifier
            'PERMCO',      # Company identifier
            'CUSIP',
            'Ticker',
            'SICCD',
            'NAICS',
            'MthCalDt',    # Date
            'MthRet',      # Target variable (monthly return)
            'sprtrn'       # Market benchmark
        ]
        
        # Check which essential columns are actually in the dataset
        available_columns = [col for col in essential_columns if col in self.data.columns]
        
        # Select only the available essential columns
        self.data = self.data[available_columns]
        
        print(f"Selected {len(available_columns)} essential columns: {', '.join(available_columns)}")
        
        return self

    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset based on their importance and frequency.

        For Market Returns (sprtrn):

        Very few missing values (0.01%)
        Since market returns are the same for all stocks on a given date, we can fill missing values by looking at other stocks on the same date
        If an entire date is missing, we use time-series interpolation (forward/backward fill)
        For CUSIP:

        Moderate number of missing values (11.03%)
        CUSIP should be consistent for a given security (PERMNO)
        We fill missing values within each security group using forward and backward fill
        Some securities might have no valid CUSIP at all, but this is acceptable since PERMNO is our primary identifier
        For Monthly Returns (MthRet):

        This is our target variable
        For a prediction task, we generally should not impute target values
        We remove rows with missing return values (1.61% of data)
        This is a standard approach for handling missing target values in supervised learning
        
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """


        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        original_rows = len(self.data)
        
        # 1. Handle missing market returns (sprtrn) - very few missing values
        # For market returns, we can forward fill or backward fill since it's the same for all stocks on a given date
        if self.market_return_column in self.data.columns:
            missing_sprtrn = self.data[self.market_return_column].isnull().sum()
            if missing_sprtrn > 0:
                # Group by date and fill missing market returns
                market_returns = self.data[[self.date_column, self.market_return_column]].drop_duplicates()
                market_returns = market_returns.sort_values(self.date_column)
                # Fill using both forward and backward fill to handle edge cases
                market_returns[self.market_return_column] = market_returns[self.market_return_column].fillna(method='ffill').fillna(method='bfill')
                
                # Merge back with original data
                self.data = self.data.drop(columns=[self.market_return_column])
                self.data = pd.merge(
                    self.data, 
                    market_returns, 
                    on=self.date_column, 
                    how='left'
                )
                print(f"Filled {missing_sprtrn} missing market returns using time-series interpolation.")
        
        # 2. Handle missing CUSIP - not critical for prediction but useful for merging
        # We can forward fill within each PERMNO group since CUSIP should be consistent for a security
        if 'CUSIP' in self.data.columns:
            missing_cusip = self.data['CUSIP'].isnull().sum()
            if missing_cusip > 0:
                self.data['CUSIP'] = self.data.groupby(self.id_column)['CUSIP'].fillna(method='ffill')
                self.data['CUSIP'] = self.data.groupby(self.id_column)['CUSIP'].fillna(method='bfill')
                remaining_missing = self.data['CUSIP'].isnull().sum()
                print(f"Filled {missing_cusip - remaining_missing} missing CUSIP values within security groups.")
                if remaining_missing > 0:
                    print(f"Note: {remaining_missing} CUSIP values remain missing for securities with no valid CUSIP.")
        
        # 3. Handle missing returns (MthRet) - this is our target variable
        # For a prediction task, we should generally remove rows with missing target values
        missing_returns = self.data[self.return_column].isnull().sum()
        if missing_returns > 0:
            self.data = self.data.dropna(subset=[self.return_column])
            print(f"Removed {missing_returns} rows with missing return values.")
        
        # Report the results
        final_rows = len(self.data)
        rows_removed = original_rows - final_rows
        print(f"Total rows removed: {rows_removed} ({rows_removed/original_rows:.2%} of original data)")
        
        return self

    def sort_by_date(self, ascending=True):
        """
        Sort the dataset by date.
        
        Parameters:
        -----------
        ascending : bool, default=True
            If True, sort from oldest to newest (chronological order).
            If False, sort from newest to oldest (reverse chronological order).
            
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Ensure date column is in datetime format
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        # Sort by date first, then by security identifier
        self.data = self.data.sort_values(
            [self.date_column, self.id_column], 
            ascending=[ascending, True]
        )
        
        date_range = f"from {self.data[self.date_column].min().strftime('%Y-%m-%d')} to {self.data[self.date_column].max().strftime('%Y-%m-%d')}"
        order = "chronological" if ascending else "reverse chronological"
        print(f"Data sorted in {order} order {date_range}")
        
        return self

    def filter_by_date(self, start_date=None, end_date=None):
        """
        Filter the dataset to include only data within a specified date range.
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            The earliest date to include in the dataset (inclusive).
            Format: 'YYYY-MM-DD' or datetime object.
            If None, no lower bound is applied.
        
        end_date : str or datetime, optional
            The latest date to include in the dataset (inclusive).
            Format: 'YYYY-MM-DD' or datetime object.
            If None, no upper bound is applied.
            
        Returns:
        --------
        self : ReturnPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Ensure date column is in datetime format
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        original_rows = len(self.data)
        
        # Apply start date filter if provided
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data[self.date_column] >= start_date]
            print(f"Removed data before {start_date.strftime('%Y-%m-%d')}")
        
        # Apply end date filter if provided
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            self.data = self.data[self.data[self.date_column] <= end_date]
            print(f"Removed data after {end_date.strftime('%Y-%m-%d')}")
        
        # Report the results
        final_rows = len(self.data)
        rows_removed = original_rows - final_rows
        
        if rows_removed > 0:
            print(f"Total rows removed: {rows_removed} ({rows_removed/original_rows:.2%} of original data)")
            print(f"Remaining date range: {self.data[self.date_column].min().strftime('%Y-%m-%d')} to {self.data[self.date_column].max().strftime('%Y-%m-%d')}")
            print(f"Remaining securities: {self.data[self.id_column].nunique()}")
        else:
            print("No data was filtered out based on the provided date range.")
        
        return self


