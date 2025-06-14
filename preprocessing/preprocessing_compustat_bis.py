import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class CompustatPreprocessor:
    """
    A class for preprocessing Compustat quarterly financial data.

    This class provides methods to load, clean, and prepare Compustat data
    for machine learning applications in finance.
    """

    def __init__(self, date_column='datadate', id_column='cusip'):
        """
        Initialize the CompustatPreprocessor with column names.

        Parameters:
        -----------
        date_column : str
            Column name for the date
        id_column : str
            Column name for the stock identifier
        """
        self.date_column = date_column
        self.id_column = id_column
        self.data = None
        self.original_shape = None

        # Define essential identifier columns
        self.id_columns = ['gvkey', 'cusip', 'tic', 'conm', 'cik', 'exchg']

        # Define key financial statement columns based on Compustat documentation
        # Income Statement Items
        self.income_statement_columns = [
            'revtq',    # Revenue/Sales
            'saleq',    # Net Sales
            'cogsq',    # Cost of Goods Sold
            'xsgaq',    # Selling, General & Administrative Expenses
            'oibdpq',   # Operating Income Before Depreciation
            'dpq',      # Depreciation and Amortization
            'oiadpq',   # Operating Income After Depreciation
            'niq',      # Net Income
            'ibq',      # Income Before Extraordinary Items
            'piq',      # Pretax Income
            'txtq',     # Income Taxes
            'xintq',    # Interest Expense
            'epsfxq',   # Earnings Per Share (Basic) Excluding Extraordinary Items
            'epspxq'    # Earnings Per Share (Diluted) Excluding Extraordinary Items
        ]

        # Balance Sheet Items
        self.balance_sheet_columns = [
            'atq',      # Total Assets
            'ltq',      # Total Liabilities
            'dlttq',    # Long-term Debt
            'dlcq',     # Debt in Current Liabilities
            'cheq',     # Cash and Short-Term Investments
            'rectq',    # Receivables
            'invtq',    # Inventory
            'ppentq',   # Property, Plant & Equipment (Net)
            'seqq',     # Shareholders' Equity
            'ceqq',     # Common Equity
            'pstkq',    # Preferred Stock
            'actq',     # Current Assets
            'lctq'      # Current Liabilities
        ]

        # Cash Flow Items
        self.cash_flow_columns = [
            'oancfq',   # Operating Activities Net Cash Flow
            'capxq',    # Capital Expenditures
            'dvpq',     # Cash Dividends
            'ivncfq',   # Investing Activities Net Cash Flow
            'fincfq'    # Financing Activities Net Cash Flow
        ]

        # Market-Related Items
        self.market_columns = [
            'cshoq',    # Common Shares Outstanding
            'prccq',    # Price Close - Quarter
            'mkvaltq'   # Market Value
        ]

    def load(self, file_path, skiprows=None, nrows=None, columns=None, header=0):
        """
        Load the Compustat dataset from a CSV or Parquet file with options for row ranges.

        Parameters:
        -----------
        file_path : str
            Path to the CSV or Parquet file
        skiprows : int, optional
            Number of rows to skip from the beginning of the file
        nrows : int, optional
            Number of rows to load (useful for large datasets)
        columns : list, optional
            Specific columns to load (if None, loads all columns)
        header : int, optional
            Row to use as column names (0-indexed)

        Returns:
        --------
        self : CompustatChunkProcessor
            Returns self for method chaining
        """
        try:
            print(f"Loading data from {file_path}")
            if skiprows:
                print(f"Skipping first {skiprows} rows")
            if nrows:
                print(f"Loading {nrows} rows")
            if columns:
                print(f"Loading {len(columns)} specific columns")

            if file_path.endswith('.csv'):
                # For CSV files, we can use skiprows and nrows directly
                if skiprows is not None:
                    # When skipping rows, we need to read the header first
                    header_df = pd.read_csv(file_path, nrows=1)
                    column_names = header_df.columns.tolist()

                    # Then read the actual data with the column names
                    self.data = pd.read_csv(
                        file_path,
                        skiprows=skiprows+1,  # +1 to account for header
                        nrows=nrows,
                        usecols=columns,
                        names=column_names  # Use the column names from the header
                    )
                else:
                    # Normal case without skipping rows
                    self.data = pd.read_csv(
                        file_path,
                        nrows=nrows,
                        usecols=columns
                    )
            elif file_path.endswith('.parquet'):
                # For Parquet files, we need to handle row ranges differently
                if columns:
                    # Load only specified columns
                    self.data = pd.read_parquet(file_path, columns=columns)
                else:
                    # Load all columns
                    self.data = pd.read_parquet(file_path)

                # Apply row range filtering
                if skiprows is not None or nrows is not None:
                    start_idx = skiprows if skiprows is not None else 0
                    end_idx = start_idx + nrows if nrows is not None else len(self.data)
                    self.data = self.data.iloc[start_idx:end_idx]
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")

            print(f"Loaded dataset with shape: {self.data.shape}")

            # Convert date column to datetime if it exists
            if self.date_column in self.data.columns:
                try:
                    self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
                    print(f"Converted {self.date_column} to datetime")
                except Exception as e:
                    print(f"Warning: Could not convert {self.date_column} to datetime: {e}")

            return self
        except Exception as e:
            print(f"Error {e} loading dataset")

    def explore_data(self, show_dtypes=True, show_missing=True, show_unique=True, save_missing_values=True):
        """
        Explore the dataset to understand its structure and identify issues.

        Parameters:
        -----------
        show_dtypes : bool
            Whether to show data types of columns
        show_missing : bool
            Whether to show missing value statistics
        show_unique : bool
            Whether to show unique value counts for categorical columns

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

        # Data types
        if show_dtypes:
            print("\n=== Data Types ===")
            dtypes_df = pd.DataFrame({
                'Column': self.data.dtypes.index,
                'Type': self.data.dtypes.values
            })
            print(dtypes_df)

        # Missing values
        if show_missing:
            missing = self.data.isnull().sum()
            missing_pct = (missing / len(self.data)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Values': missing.values,
                'Percentage': missing_pct.values
            }).sort_values('Missing Values', ascending=False)

            print("\n=== Missing Values ===")
            print(missing_df[missing_df['Missing Values'] > 0])
            print(f"Total columns with missing values: {sum(missing > 0)}")

        # Unique values for categorical columns
        if show_unique:
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                print("\n=== Categorical Columns ===")
                for col in categorical_cols:
                    unique_values = self.data[col].nunique()
                    print(f"{col}: {unique_values} unique values")
                    if unique_values < 10:  # Show all values if there aren't too many
                        print(f"Values: {sorted(self.data[col].unique())}")

        # Time range
        if self.date_column in self.data.columns:
            try:
                date_range = pd.to_datetime(self.data[self.date_column])
                print("\n=== Time Range ===")
                print(f"From {date_range.min()} to {date_range.max()}")
            except:
                print(f"Could not convert {self.date_column} to datetime.")

        # Company count
        if self.id_column in self.data.columns:
            company_count = self.data[self.id_column].nunique()
            print(f"\n=== Company Count ===")
            print(f"Dataset contains {company_count} unique companies")

        return {
            'shape': self.data.shape,
            'missing': missing_df if show_missing else None,
            'company_count': company_count if self.id_column in self.data.columns else None
        }

    def convert_dates(self):
        """
        Convert date column to datetime format.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            try:
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
                print(f"Converted {self.date_column} to datetime.")
            except Exception as e:
                print(f"Error converting {self.date_column} to datetime: {e}")
                raise

        return self

    def select_important_columns(self):
        """
        Select important financial columns with good coverage for sentiment analysis.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Core identifiers (all with excellent coverage)
        identifiers = [
            self.date_column,  # Date first (100% coverage)
            'gvkey',           # Global company key (100% coverage)
            'cusip',           # CUSIP identifier (100% coverage)
            'tic',             # Ticker symbol (99.99% coverage)
            'conm',            # Company name (100% coverage)
            'exchg',           # Exchange code (100% coverage)
            'cik'              # Central Index Key (88.22% coverage) - useful for SEC filings
        ]

        # Financial Performance Metrics (>70% coverage)
        performance_metrics = [
            'epspxy',          # Earnings Per Share (Diluted) Excluding Extraordinary Items (81.92% coverage)
            'epspiy',          # Earnings Per Share (Diluted) Including Extraordinary Items (81.84% coverage)
            'epsfiy',          # Earnings Per Share (Basic) Including Extraordinary Items (78.95% coverage)
            'epsfxy',          # Earnings Per Share (Basic) Excluding Extraordinary Items (78.80% coverage)
            'oiadpy',          # Operating Income After Depreciation (75.22% coverage)
            'cshpry'           # Common Shares Used to Calculate EPS (83.34% coverage)
        ]

        # Additional Financial Metrics (>50% coverage)
        additional_metrics = [
            'dvy',             # Cash Dividends (56.14% coverage)
            'capxy',           # Capital Expenditures (55.53% coverage)
            'sstky',           # Sale of Common and Preferred Stock (54.93% coverage)
            'aqcy',            # Acquisitions (54.84% coverage)
            'prstkcy',         # Purchase of Common and Preferred Stock (54.64% coverage)
            'dltry',           # Long-Term Debt Reduction (54.28% coverage)
            'dltisy',          # Long-Term Debt Issuance (54.14% coverage)
            'sivy',            # Short-Term Investments - Change (53.47% coverage)
            'ivchy',           # Inventory - Change (53.47% coverage)
            'chechy',          # Cash and Cash Equivalents - Change (52.90% coverage)
            'ibcy',            # Income Before Extraordinary Items (Cash Flow) (52.36% coverage)
            'fopoy',           # Foreign Exchange Income (Loss) (52.34% coverage)
            'dpcy'             # Depreciation and Amortization (50.84% coverage)
        ]

        # Income Statement Items (>45% coverage)
        income_items = [
            'niy',             # Net Income (48.93% coverage)
            'saley',           # Net Sales/Revenue (48.85% coverage)
            'iby',             # Income Before Extraordinary Items (49.00% coverage)
            'revty',           # Revenue (47.40% coverage)
            'txty',            # Income Taxes (48.26% coverage)
            'piy',             # Pretax Income (48.17% coverage)
            'txdcy',           # Deferred Taxes (47.39% coverage)
            'nopiy',           # Non-Operating Income (45.96% coverage)
            'xopry',           # Operating Expenses (44.99% coverage)
            'cogsy'            # Cost of Goods Sold (45.10% coverage)
        ]

        # Combine all important columns
        important_columns = identifiers + performance_metrics + additional_metrics + income_items

        # Filter to only include columns that actually exist in the dataset
        available_columns = [col for col in important_columns if col in self.data.columns]

        if not available_columns:
            print("Warning: None of the predefined important columns exist in the dataset.")
            return self

        # Keep only the important columns
        original_cols = self.data.columns.tolist()
        self.data = self.data[available_columns]

        print(f"Selected {len(available_columns)} important columns out of {len(original_cols)}")
        print(f"First few columns: {', '.join(available_columns[:5])}")

        # Print column categories
        print("\n=== Selected Column Categories ===")
        print(f"Identifiers: {', '.join([col for col in identifiers if col in available_columns])}")
        print(f"Performance Metrics: {', '.join([col for col in performance_metrics if col in available_columns])}")
        print(f"Additional Financial Metrics: {', '.join([col for col in additional_metrics if col in available_columns])}")
        print(f"Income Statement Items: {', '.join([col for col in income_items if col in available_columns])}")

        return self

    def global_sort_by_date(self):
        """
        Sort the entire dataset by date from oldest to newest.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            try:
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            except Exception as e:
                print(f"Error converting {self.date_column} to datetime: {e}")
                raise

        # Sort by date only (not by company)
        self.data = self.data.sort_values(self.date_column)

        print(f"Data sorted globally by {self.date_column} from oldest to newest")

        return self

    def handle_missing_values(self, method='drop_cols', threshold=0.5):
        """
        Handle missing values in the dataset.

        Parameters:
        -----------
        method : str
            Method to handle missing values ('drop_cols', 'drop_rows', 'ffill', 'zero', or 'empty_strings')
        threshold : float
            Maximum percentage of missing values allowed for a column (for 'drop_cols')

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        print(f"Handling missing values using {method} method...")

        # Check for missing values
        total_cells = self.data.size
        missing_cells = self.data.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100

        print(f"Missing values: {missing_cells} out of {total_cells} cells ({missing_pct:.2f}%)")

        if missing_cells == 0:
            print("No missing values found.")
            return self

        # Store original shape for reporting
        before_shape = self.data.shape

        # Handle missing values based on method
        if method == 'drop_rows':
            # Drop rows with any missing values
            self.data = self.data.dropna()

        elif method == 'drop_cols':
            # Drop columns with too many missing values
            missing_by_col = self.data.isna().mean()
            drop_cols = missing_by_col[missing_by_col > threshold].index.tolist()

            if drop_cols:
                print(f"Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing values")
                self.data = self.data.drop(columns=drop_cols)

            # For remaining columns, replace NaN with empty string in string columns
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    self.data[col] = self.data[col].fillna('')
                    print(f"Replaced NaN values with empty strings in column '{col}'")

        elif method == 'ffill':
            # Forward fill missing values within each company
            self.data = self.data.groupby(self.id_column).apply(
                lambda group: group.sort_values(self.date_column).fillna(method='ffill')
            ).reset_index(drop=True)

            # Handle any remaining missing values with backward fill
            self.data = self.data.groupby(self.id_column).apply(
                lambda group: group.sort_values(self.date_column).fillna(method='bfill')
            ).reset_index(drop=True)

            # For any remaining NaNs in string columns, replace with empty string
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    self.data[col] = self.data[col].fillna('')

        elif method == 'zero':
            # Fill missing values with zero
            self.data = self.data.fillna(0)

        elif method == 'empty_strings':
            # Replace NaNs with empty strings in string columns, keep NaNs in numeric columns
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    self.data[col] = self.data[col].fillna('')
                    print(f"Replaced NaN values with empty strings in column '{col}'")

        else:
            raise ValueError("Invalid method. Use 'drop_cols', 'drop_rows', 'ffill', 'zero', or 'empty_strings'.")

        # Report results
        after_shape = self.data.shape
        rows_removed = before_shape[0] - after_shape[0]
        cols_removed = before_shape[1] - after_shape[1]

        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing values.")
        if cols_removed > 0:
            print(f"Removed {cols_removed} columns with missing values.")

        # Check if any missing values remain
        remaining_missing = self.data.isna().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain after processing")
        else:
            print("All missing values have been handled")

        return self

    def clean_string_nan_values(self):
        """
        Clean string 'nan' values from the dataset.

        This function specifically targets string columns containing the literal 'nan'
        and replaces them with empty strings.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Count before cleaning
        string_nan_count = 0

        # For each column, check if it contains string data
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Count occurrences of string 'nan'
                col_nan_count = (self.data[col] == 'nan').sum()
                if col_nan_count > 0:
                    string_nan_count += col_nan_count
                    # Replace string 'nan' with empty string
                    self.data[col] = self.data[col].replace('nan', '')
                    print(f"Replaced {col_nan_count} string 'nan' values with empty strings in column '{col}'")

        print(f"Total string 'nan' values replaced: {string_nan_count}")

        return self


    def handle_duplicates(self):
        """
        Handle duplicate rows in the dataset.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Check for duplicates based on company ID and date
        duplicate_keys = self.data.duplicated(subset=[self.id_column, self.date_column], keep=False)
        duplicate_count = duplicate_keys.sum()

        if duplicate_count == 0:
            print("No duplicate rows found.")
            return self

        print(f"Found {duplicate_count} duplicate rows based on {self.id_column} and {self.date_column}")

        # Keep the first occurrence of each duplicate
        self.data = self.data.drop_duplicates(subset=[self.id_column, self.date_column], keep='first')

        print(f"Removed {duplicate_count - self.data.duplicated(subset=[self.id_column, self.date_column], keep=False).sum()} duplicate rows")

        return self

    def fix_data_types(self):
        """
        Fix data types for columns in the dataset and clean string 'nan' values.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        print("Fixing data types...")

        # Convert date column to datetime
        if self.date_column in self.data.columns and not pd.api.types.is_datetime64_dtype(self.data[self.date_column]):
            try:
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
                print(f"Converted {self.date_column} to datetime")
            except:
                print(f"Could not convert {self.date_column} to datetime")

        # Convert numeric columns to float
        numeric_cols = (
            self.balance_sheet_columns +
            self.income_statement_columns +
            self.cash_flow_columns +
            self.market_columns
        )

        for col in numeric_cols:
            if col in self.data.columns and not pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"Converted {col} to numeric")
                except:
                    print(f"Could not convert {col} to numeric")

        # Convert ID columns to string and clean 'nan' values
        for col in self.id_columns:
            if col in self.data.columns:
                # Replace string 'nan' with empty string before type conversion
                if self.data[col].dtype == 'object':
                    nan_count = (self.data[col] == 'nan').sum()
                    if nan_count > 0:
                        self.data[col] = self.data[col].replace('nan', '')
                        print(f"Replaced {nan_count} string 'nan' values in column '{col}'")

                # Convert to string type
                if not pd.api.types.is_string_dtype(self.data[col]):
                    self.data[col] = self.data[col].astype(str)
                    print(f"Converted {col} to string")

        # Clean 'nan' strings in all remaining string columns
        string_cols = self.data.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in self.id_columns:  # Skip columns we already processed
                nan_count = (self.data[col] == 'nan').sum()
                if nan_count > 0:
                    self.data[col] = self.data[col].replace('nan', '')
                    print(f"Replaced {nan_count} string 'nan' values in column '{col}'")

        return self

    def create_financial_ratios(self):
        """
        Create common financial ratios from the Compustat data.

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        print("Creating basic financial ratios...")

        # Helper function to safely calculate ratios
        def safe_divide(a, b):
            if isinstance(a, pd.Series) and isinstance(b, pd.Series):
                return a / b.replace(0, np.nan)
            elif isinstance(b, (int, float)) and b == 0:
                return np.nan
            else:
                return a / b if b != 0 else np.nan

        # Profitability Ratios
        if all(col in self.data.columns for col in ['niq', 'atq']):
            self.data['roa'] = safe_divide(self.data['niq'], self.data['atq'])  # Return on Assets
            print("Created ROA (Return on Assets)")

        if all(col in self.data.columns for col in ['niq', 'ceqq']):
            self.data['roe'] = safe_divide(self.data['niq'], self.data['ceqq'])  # Return on Equity
            print("Created ROE (Return on Equity)")

        if all(col in self.data.columns for col in ['oiadpq', 'saleq']):
            self.data['operating_margin'] = safe_divide(self.data['oiadpq'], self.data['saleq'])  # Operating Margin
            print("Created Operating Margin")

        if all(col in self.data.columns for col in ['niq', 'saleq']):
            self.data['net_margin'] = safe_divide(self.data['niq'], self.data['saleq'])  # Net Profit Margin
            print("Created Net Profit Margin")

        # Liquidity Ratios
        if all(col in self.data.columns for col in ['actq', 'lctq']):
            self.data['current_ratio'] = safe_divide(self.data['actq'], self.data['lctq'])  # Current Ratio
            print("Created Current Ratio")

        if all(col in self.data.columns for col in ['actq', 'invtq', 'lctq']):
            self.data['quick_ratio'] = safe_divide(self.data['actq'] - self.data['invtq'], self.data['lctq'])  # Quick Ratio
            print("Created Quick Ratio")

        # Leverage Ratios
        if all(col in self.data.columns for col in ['ltq', 'atq']):
            self.data['debt_to_assets'] = safe_divide(self.data['ltq'], self.data['atq'])  # Debt to Assets
            print("Created Debt to Assets")

        if all(col in self.data.columns for col in ['ltq', 'ceqq']):
            self.data['debt_to_equity'] = safe_divide(self.data['ltq'], self.data['ceqq'])  # Debt to Equity
            print("Created Debt to Equity")

        # Valuation Metrics
        if all(col in self.data.columns for col in ['ceqq', 'cshoq', 'prccq']):
            market_cap = self.data['cshoq'] * self.data['prccq']
            self.data['book_to_market'] = safe_divide(self.data['ceqq'], market_cap)
            print("Created Book-to-Market")

        return self

    def analyze_column_variability(self, top_n=30):
        """
        Analyze the variability of columns to identify which ones might be useful.

        Parameters:
        -----------
        top_n : int
            Number of columns with the highest variability to display

        Returns:
        --------
        pandas.DataFrame
            DataFrame with column variability statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns

        # Calculate statistics for each column
        stats = []
        for col in numeric_cols:
            # Skip columns with too many missing values
            missing_pct = self.data[col].isna().mean() * 100
            if missing_pct > 90:
                continue

            # Calculate basic statistics
            unique_count = self.data[col].nunique()
            unique_pct = (unique_count / len(self.data)) * 100

            # Calculate coefficient of variation (if possible)
            try:
                mean = self.data[col].mean()
                std = self.data[col].std()
                cv = (std / mean) if mean != 0 else np.nan
            except:
                cv = np.nan

            stats.append({
                'Column': col,
                'Missing (%)': missing_pct,
                'Unique Values': unique_count,
                'Unique (%)': unique_pct,
                'Mean': self.data[col].mean(),
                'Std Dev': self.data[col].std(),
                'CV': cv
            })

        # Convert to DataFrame and sort by variability
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('CV', ascending=False)

        print(f"\n=== Top {top_n} Columns by Variability ===")
        print(stats_df.head(top_n))

        return stats_df

    def get_data(self):
        """
        Get the processed data.

        Returns:
        --------
        pandas.DataFrame
            The processed dataset
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        return self.data.copy()

    def export_data(self, file_path, format='parquet'):
        """
        Export the data to a file.

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
            raise ValueError("No data loaded. Call load() first.")

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
        print(f"Shape: {self.data.shape}")

        return file_path

    def plot_missing_values(self, top_n=30):
        """
        Plot missing values in the dataset.

        Parameters:
        -----------
        top_n : int
            Number of columns with the most missing values to plot

        Returns:
        --------
        None
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Calculate missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100

        # Create a DataFrame for plotting
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': missing_pct.values
        }).sort_values('Missing Values', ascending=False)

        # Filter to columns with missing values
        missing_df = missing_df[missing_df['Missing Values'] > 0]

        if missing_df.empty:
            print("No missing values to plot.")
            return

        # Take top N columns with most missing values
        plot_df = missing_df.head(top_n)

        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(plot_df['Column'], plot_df['Percentage'])

        # Add percentage labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f"{plot_df['Percentage'].iloc[i]:.1f}%",
                va='center'
            )

        plt.title('Percentage of Missing Values by Column')
        plt.xlabel('Percentage Missing')
        plt.ylabel('Column')
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()

    def check_column_coverage(self):
        """
        Check the coverage (non-missing percentage) of columns in the dataset.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with column coverage statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Calculate coverage for each column
        coverage = []
        for col in self.data.columns:
            missing = self.data[col].isna().sum()
            missing_pct = (missing / len(self.data)) * 100
            coverage_pct = 100 - missing_pct

            coverage.append({
                'Column': col,
                'Coverage (%)': coverage_pct,
                'Missing (%)': missing_pct,
                'Data Type': self.data[col].dtype
            })

        # Convert to DataFrame and sort by coverage
        coverage_df = pd.DataFrame(coverage)
        coverage_df = coverage_df.sort_values('Coverage (%)', ascending=False)

        print("\n=== Column Coverage ===")
        print(coverage_df)

        return coverage_df

    def recommend_columns_for_sentiment(self, min_coverage=50):
        """
        Recommend columns that are most useful for sentiment analysis based on coverage.

        Parameters:
        -----------
        min_coverage : float
            Minimum coverage percentage to recommend a column

        Returns:
        --------
        list
            List of recommended columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Calculate coverage for each column
        coverage_df = self.check_column_coverage()

        # Filter columns with good coverage
        good_coverage = coverage_df[coverage_df['Coverage (%)'] >= min_coverage]

        # Always include key identifier columns
        essential_columns = ['datadate', 'gvkey', 'cusip', 'tic', 'conm', 'exchg', 'cik']
        essential_available = [col for col in essential_columns if col in self.data.columns]

        # Categorize financial columns by type
        earnings_cols = ['epspxy', 'epspiy', 'epsfxy', 'epsfiy', 'niy', 'iby', 'ibcy', 'oiadpy']
        revenue_cols = ['saley', 'revty']
        expense_cols = ['cogsy', 'xopry', 'xsgay', 'dpcy']
        cash_flow_cols = ['capxy', 'dvy', 'oancfy', 'fincfy', 'ivncfy']
        debt_cols = ['dltisy', 'dltry', 'dlcchy']
        asset_cols = ['chechy', 'ivchy', 'sivy', 'aqcy']
        equity_cols = ['sstky', 'prstkcy']

        # Filter columns by category and coverage
        earnings_available = [col for col in earnings_cols if col in good_coverage['Column'].tolist()]
        revenue_available = [col for col in revenue_cols if col in good_coverage['Column'].tolist()]
        expense_available = [col for col in expense_cols if col in good_coverage['Column'].tolist()]
        cash_flow_available = [col for col in cash_flow_cols if col in good_coverage['Column'].tolist()]
        debt_available = [col for col in debt_cols if col in good_coverage['Column'].tolist()]
        asset_available = [col for col in asset_cols if col in good_coverage['Column'].tolist()]
        equity_available = [col for col in equity_cols if col in good_coverage['Column'].tolist()]

        # Combine all recommended columns
        recommended = (
            essential_available +
            earnings_available +
            revenue_available +
            expense_available +
            cash_flow_available +
            debt_available +
            asset_available +
            equity_available
        )

        # Remove duplicates while preserving order
        recommended = list(dict.fromkeys(recommended))

        print(f"\n=== Recommended Columns for Sentiment Analysis ({len(recommended)}) ===")
        print(f"Essential: {', '.join(essential_available)}")
        print(f"Earnings: {', '.join(earnings_available)}")
        print(f"Revenue: {', '.join(revenue_available)}")
        print(f"Expenses: {', '.join(expense_available)}")
        print(f"Cash Flow: {', '.join(cash_flow_available)}")
        print(f"Debt: {', '.join(debt_available)}")
        print(f"Assets: {', '.join(asset_available)}")
        print(f"Equity: {', '.join(equity_available)}")

        return recommended

    def create_sentiment_indicators(self):
      """
      Create financial indicators that are particularly useful for sentiment analysis.
      This version is designed to work after interpolation to monthly frequency.

      Returns:
      --------
      self : CompustatPreprocessor
          Returns self for method chaining
      """
      if self.data is None:
          raise ValueError("No data loaded. Call load() first.")

      print("Creating financial indicators for sentiment analysis...")

      # Helper function to safely calculate ratios
      def safe_divide(a, b):
          if isinstance(a, pd.Series) and isinstance(b, pd.Series):
              return a / b.replace(0, np.nan)
          elif isinstance(b, (int, float)) and b == 0:
              return np.nan
          else:
              return a / b if b != 0 else np.nan

      # Process each company separately to avoid the length mismatch issue
      company_groups = []
      company_count = self.data[self.id_column].nunique()
      print(f"Processing {company_count} companies...")

      # Track progress
      processed = 0

      # Group by company ID
      for company_id, company_data in self.data.groupby(self.id_column):
          # Sort by date
          company_data = company_data.sort_values(self.date_column)

          # 1. Earnings Growth (YoY) - Strong indicator of company performance
          if 'niy' in company_data.columns:
              # For monthly data, shift by 12 months instead of 4 quarters
              company_data['niy_prev_year'] = company_data['niy'].shift(12)
              company_data['earnings_growth'] = safe_divide(
                  company_data['niy'] - company_data['niy_prev_year'],
                  company_data['niy_prev_year'].abs()
              )

          # 2. Revenue Growth (YoY) - Important for top-line growth sentiment
          if 'saley' in company_data.columns:
              company_data['saley_prev_year'] = company_data['saley'].shift(12)
              company_data['revenue_growth'] = safe_divide(
                  company_data['saley'] - company_data['saley_prev_year'],
                  company_data['saley_prev_year'].abs()
              )

          # 3. Debt Change - Increasing debt can be a negative sentiment signal
          if 'dltisy' in company_data.columns and 'dltry' in company_data.columns:
              company_data['net_debt_change'] = company_data['dltisy'] - company_data['dltry']

          # 4. Cash Flow to Earnings - Quality of earnings indicator
          if 'oancfy' in company_data.columns and 'niy' in company_data.columns:
              company_data['cf_to_earnings'] = safe_divide(company_data['oancfy'], company_data['niy'])

          # 5. Capital Expenditure Trend - Investment in future growth
          if 'capxy' in company_data.columns:
              company_data['capxy_prev_year'] = company_data['capxy'].shift(12)
              company_data['capex_growth'] = safe_divide(
                  company_data['capxy'] - company_data['capxy_prev_year'],
                  company_data['capxy_prev_year'].abs()
              )

          # 6. Earnings Surprise - Important for market reaction
          if 'epspxy' in company_data.columns:
              # For monthly data, use 1-month shift for quarter-over-quarter comparison
              company_data['epspxy_prev_qtr'] = company_data['epspxy'].shift(3)  # 3 months = 1 quarter
              company_data['eps_surprise'] = company_data['epspxy'] - company_data['epspxy_prev_qtr']

          # 7. Dividend Changes - Signal of financial health
          if 'dvy' in company_data.columns:
              company_data['dvy_prev_year'] = company_data['dvy'].shift(12)
              company_data['dividend_change'] = safe_divide(
                  company_data['dvy'] - company_data['dvy_prev_year'],
                  company_data['dvy_prev_year'].abs()
              )

          # 8. Stock Repurchase Intensity - Signal of management confidence
          if 'prstkcy' in company_data.columns and 'saley' in company_data.columns:
              company_data['repurchase_intensity'] = safe_divide(company_data['prstkcy'], company_data['saley'])

          # Add the processed company data to our list
          company_groups.append(company_data)

          # Update progress
          processed += 1
          if processed % 100 == 0:
              print(f"Processed {processed}/{company_count} companies ({processed/company_count*100:.1f}%)")

      # Combine all company data
      self.data = pd.concat(company_groups, ignore_index=True)

      # Drop temporary columns used for calculations
      temp_cols = [col for col in self.data.columns if col.endswith('_prev_year') or col.endswith('_prev_qtr')]
      if temp_cols:
          self.data = self.data.drop(columns=temp_cols)
          print(f"Removed {len(temp_cols)} temporary calculation columns")

      print("Created financial sentiment indicators")
      return self


    def interpolate_quarterly_to_monthly(self, min_coverage=30, min_data_points=3, min_valid_columns=2):
      """
      Convert quarterly Compustat data to monthly frequency using data type-specific methods.

      This function primarily uses forward fill (ffill) for financial data to maintain the
      step-like nature of financial reporting, with optional specialized handling for
      specific data types.

      Parameters:
      -----------
      min_coverage : float
          Minimum coverage percentage required for a column to be interpolated
      min_data_points : int
          Minimum number of valid data points required for a company to be interpolated
      min_valid_columns : int
          Minimum number of valid numeric columns required for a company to be included

      Returns:
      --------
      self : CompustatPreprocessor
          Returns self for method chaining
      """
      if self.data is None:
          raise ValueError("No data loaded. Call load() first.")

      print("Converting quarterly data to monthly frequency using forward fill...")

      # Check column coverage to determine which columns to process
      coverage_df = self.check_column_coverage()

      # Filter columns with sufficient coverage
      good_coverage_cols = coverage_df[coverage_df['Coverage (%)'] >= min_coverage]['Column'].tolist()

      # Always include essential identifier columns
      essential_cols = ['datadate', 'gvkey', 'cusip', 'tic', 'conm', 'exchg', 'cik']
      essential_available = [col for col in essential_cols if col in self.data.columns]

      # Identify numeric columns for processing
      numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()

      # Columns to process: numeric columns with good coverage
      cols_to_process = [col for col in numeric_cols if col in good_coverage_cols]

      # Columns to keep without processing: essential columns and non-numeric columns
      cols_to_keep = list(set(essential_available) - set(cols_to_process))

      print(f"Found {len(cols_to_process)} numeric columns with ≥{min_coverage}% coverage to process")
      print(f"Keeping {len(cols_to_keep)} essential/non-numeric columns without processing")

      # Categorize columns by financial statement type for specialized handling
      balance_sheet_cols = [col for col in self.balance_sheet_columns if col in cols_to_process]
      income_statement_cols = [col for col in self.income_statement_columns if col in cols_to_process]
      cash_flow_cols = [col for col in self.cash_flow_columns if col in cols_to_process]
      market_cols = [col for col in self.market_columns if col in cols_to_process]

      # Other numeric columns not categorized
      other_cols = [col for col in cols_to_process
                  if col not in balance_sheet_cols
                  and col not in income_statement_cols
                  and col not in cash_flow_cols
                  and col not in market_cols]

      print(f"Column categories:")
      print(f"- Balance sheet items: {len(balance_sheet_cols)}")
      print(f"- Income statement items: {len(income_statement_cols)}")
      print(f"- Cash flow items: {len(cash_flow_cols)}")
      print(f"- Market items: {len(market_cols)}")
      print(f"- Other numeric items: {len(other_cols)}")

      # Create a new dataframe to store monthly data
      monthly_data = []

      # Process each company separately
      company_count = self.data[self.id_column].nunique()
      print(f"Processing {company_count} companies...")

      # Track progress and statistics
      processed = 0
      skipped_few_points = 0
      skipped_few_columns = 0
      included_companies = 0

      # Group by company ID
      for company_id, company_data in self.data.groupby(self.id_column):
          # Sort by date
          company_data = company_data.sort_values(self.date_column)

          # Skip if fewer than min_data_points data points
          if len(company_data) < min_data_points:
              skipped_few_points += 1
              continue

          # Check if company has enough valid numeric columns
          valid_columns = 0
          for col in cols_to_process:
              if col in company_data.columns:
                  # Count non-NaN values
                  valid_values = company_data[col].notna().sum()
                  if valid_values >= min_data_points:
                      valid_columns += 1

          # Skip if company doesn't have enough valid columns
          if valid_columns < min_valid_columns:
              skipped_few_columns += 1
              continue

          # Create a date range from first to last date with monthly frequency
          start_date = company_data[self.date_column].min()
          end_date = company_data[self.date_column].max()

          # Create monthly date range
          monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')

          # Create a dataframe with monthly dates
          company_monthly = pd.DataFrame({self.date_column: monthly_dates})

          # Add company identifier
          company_monthly[self.id_column] = company_id

          # Add other essential columns (copy from nearest quarterly data)
          for col in cols_to_keep:
              if col in company_data.columns and col != self.date_column and col != self.id_column:
                  # Use merge_asof to get the most recent value for each monthly date
                  company_monthly = pd.merge_asof(
                      company_monthly.sort_values(self.date_column),
                      company_data[[self.date_column, col]].sort_values(self.date_column),
                      on=self.date_column,
                      direction='backward'
                  )

          # Track if any columns were successfully processed
          processed_any = False

          # Process numeric columns based on their type
          for col in cols_to_process:
              if col in company_data.columns:
                  # Skip columns with too many NaNs
                  if company_data[col].isna().mean() > 0.7:  # More than 70% missing
                      continue

                  try:
                      # Create a temporary dataframe with just this column
                      temp_df = pd.DataFrame({
                          self.date_column: company_data[self.date_column],
                          col: company_data[col]
                      }).set_index(self.date_column)

                      # Reindex to monthly frequency
                      temp_monthly = temp_df.reindex(monthly_dates)

                      # Apply appropriate method based on column type
                      if col in balance_sheet_cols:
                          # For balance sheet items: use forward fill (last known value)
                          filled_values = temp_monthly[col].fillna(method='ffill')
                      elif col in income_statement_cols:
                          # For income statement: distribute quarterly values evenly across months
                          # First forward fill to get the quarterly value
                          quarterly_values = temp_monthly[col].fillna(method='ffill')
                          # Then divide by 3 to get monthly equivalent
                          filled_values = quarterly_values / 3
                      elif col in cash_flow_cols:
                          # For cash flow: distribute quarterly values evenly across months
                          quarterly_values = temp_monthly[col].fillna(method='ffill')
                          filled_values = quarterly_values / 3
                      elif col in market_cols:
                          # For market data: use linear interpolation then forward fill
                          filled_values = temp_monthly[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                      else:
                          # For other columns: use forward fill as default
                          filled_values = temp_monthly[col].fillna(method='ffill')

                      # Add processed values to monthly dataframe
                      company_monthly[col] = filled_values.values

                      # # Also add month-over-month change as a feature
                      # company_monthly[f"{col}_mom_change"] = company_monthly[col].pct_change()

                      # Mark that we successfully processed at least one column
                      processed_any = True

                  except Exception as e:
                      print(f"Error processing {col} for company {company_id}: {e}")
                      continue

          # Only include companies where we successfully processed at least one column
          if processed_any:
              # Add to the list of monthly data
              monthly_data.append(company_monthly)
              included_companies += 1
          else:
              skipped_few_columns += 1

          # Update progress
          processed += 1
          if processed % 100 == 0:
              print(f"Processed {processed}/{company_count} companies ({processed/company_count*100:.1f}%)")

      # Print statistics
      print(f"\nProcessing statistics:")
      print(f"- Total companies processed: {processed}")
      print(f"- Companies skipped (too few data points): {skipped_few_points}")
      print(f"- Companies skipped (too few valid columns): {skipped_few_columns}")
      print(f"- Companies included in final dataset: {included_companies}")

      # Combine all monthly data
      if not monthly_data:
          raise ValueError("No data could be processed. Check your data and parameters.")

      monthly_df = pd.concat(monthly_data, ignore_index=True)

      # Sort by company and date
      monthly_df = monthly_df.sort_values([self.id_column, self.date_column])

      # Update the data attribute
      self.data = monthly_df

      print(f"Conversion complete. Created monthly data with shape: {self.data.shape}")

      return self


    def filter_columns_by_coverage(self, min_coverage=30):
        """
        Filter columns based on their coverage percentage.

        Parameters:
        -----------
        min_coverage : float
            Minimum coverage percentage required for a column to be kept

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        print(f"Filtering columns with coverage < {min_coverage}%...")

        # Calculate coverage for each column
        coverage_df = self.check_column_coverage()

        # Identify columns with poor coverage
        poor_coverage_cols = coverage_df[coverage_df['Coverage (%)'] < min_coverage]['Column'].tolist()

        # Always keep essential identifier columns regardless of coverage
        essential_cols = ['datadate', 'gvkey', 'cusip', 'tic', 'conm', 'exchg', 'cik']
        poor_coverage_cols = [col for col in poor_coverage_cols if col not in essential_cols]

        if poor_coverage_cols:
            print(f"Removing {len(poor_coverage_cols)} columns with poor coverage (<{min_coverage}%):")
            print(", ".join(poor_coverage_cols[:10]) + ("..." if len(poor_coverage_cols) > 10 else ""))

            # Drop columns with poor coverage
            self.data = self.data.drop(columns=poor_coverage_cols)

            print(f"Remaining columns: {len(self.data.columns)}")
        else:
            print(f"All columns have coverage ≥{min_coverage}%")

        return self

    def plot_interpolation_example(self, company_id=None, column=None, n_samples=3):
        """
        Plot examples of the interpolation for visual inspection.

        Parameters:
        -----------
        company_id : str, optional
            Specific company ID to plot. If None, random companies are selected.
        column : str, optional
            Specific column to plot. If None, a suitable numeric column is selected.
        n_samples : int, optional
            Number of random samples to plot if company_id is None

        Returns:
        --------
        None
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # If no specific column is provided, find a suitable numeric column
        if column is None:
            numeric_cols = self.data.select_dtypes(include=['number']).columns
            # Exclude columns that are likely to be identifiers or have _mom_change suffix
            suitable_cols = [col for col in numeric_cols
                            if not col.endswith('_mom_change')
                            and col not in ['gvkey', 'exchg', 'cik']]

            if not suitable_cols:
                print("No suitable numeric columns found for plotting.")
                return

            # Choose the column with the best coverage
            column = max(suitable_cols, key=lambda col: self.data[col].notna().mean())

        # If no specific company is provided, sample random companies
        if company_id is None:
            # Get companies with sufficient data points
            company_counts = self.data.groupby(self.id_column).size()
            eligible_companies = company_counts[company_counts > 10].index.tolist()

            if not eligible_companies:
                print("No companies with sufficient data points found.")
                return

            # Sample random companies
            companies_to_plot = np.random.choice(eligible_companies, min(n_samples, len(eligible_companies)), replace=False)
        else:
            companies_to_plot = [company_id]

        # Create plots
        n_companies = len(companies_to_plot)
        fig, axes = plt.subplots(n_companies, 1, figsize=(12, 5 * n_companies))

        if n_companies == 1:
            axes = [axes]  # Make axes iterable when there's only one subplot

        for i, company in enumerate(companies_to_plot):
            # Get data for this company
            company_data = self.data[self.data[self.id_column] == company].sort_values(self.date_column)

            # Skip if no data or column not available
            if len(company_data) == 0 or column not in company_data.columns:
                print(f"No data available for company {company} and column {column}")
                continue

            # Plot the data
            ax = axes[i]
            ax.plot(company_data[self.date_column], company_data[column], 'o-', label=f'{column} (Monthly)')

            # Add month-over-month change if available
            mom_col = f"{column}_mom_change"
            if mom_col in company_data.columns:
                ax2 = ax.twinx()
                ax2.plot(company_data[self.date_column], company_data[mom_col], 'r--', label='Month-over-Month Change')
                ax2.set_ylabel('Month-over-Month Change', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

            # Add labels and title
            ax.set_title(f"Interpolated {column} for {company} ({company_data['conm'].iloc[0] if 'conm' in company_data.columns else 'Unknown'})")
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.grid(True, alpha=0.3)

            # Add legend
            lines, labels = ax.get_legend_handles_labels()
            if mom_col in company_data.columns:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')
            else:
                ax.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

        return

    def filter_by_returns_cusips(self, returns_cusips_df=None, returns_cusips_file=None, cusip_column='CUSIP'):
        """
        Filter the Compustat dataset to include only companies that have return data.

        This function takes either a DataFrame containing CUSIPs from a returns dataset
        or a file path to such a dataset, and filters the Compustat data to include
        only those CUSIPs.

        Parameters:
        -----------
        returns_cusips_df : pandas.DataFrame, optional
            DataFrame containing CUSIPs from returns data
        returns_cusips_file : str, optional
            Path to a CSV file containing CUSIPs from returns data
        cusip_column : str, default='CUSIP'
            Name of the column in returns_cusips_df that contains the CUSIPs

        Returns:
        --------
        self : CompustatPreprocessor
            Returns self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Load CUSIPs from file if provided
        if returns_cusips_df is None and returns_cusips_file is not None:
            try:
                returns_cusips_df = pd.read_csv(returns_cusips_file)
                print(f"Loaded returns CUSIPs from {returns_cusips_file}")
            except Exception as e:
                raise ValueError(f"Error loading returns CUSIPs file: {e}")

        if returns_cusips_df is None:
            raise ValueError("Either returns_cusips_df or returns_cusips_file must be provided")

        # Ensure the CUSIP column exists
        if cusip_column not in returns_cusips_df.columns:
            raise ValueError(f"CUSIP column '{cusip_column}' not found in returns data")

        # Extract unique CUSIPs from returns data
        returns_cusips = set(returns_cusips_df[cusip_column].astype(str).str.strip())

        # Count unique CUSIPs in returns data
        unique_returns_cusips = len(returns_cusips)
        print(f"Found {unique_returns_cusips} unique CUSIPs in returns data")

        # Count unique CUSIPs in Compustat data before filtering
        compustat_cusips_before = self.data[self.id_column].nunique()
        print(f"Compustat data contains {compustat_cusips_before} unique CUSIPs before filtering")

        # Handle potential CUSIP format differences
        # Some datasets use 8-digit CUSIPs, others use 9-digit CUSIPs
        # We'll try to match on the first 8 digits to be safe

        # Create a standardized CUSIP column for matching
        self.data['cusip_match'] = self.data[self.id_column].astype(str).str.strip().str[:8]
        returns_cusips_8digit = {cusip[:8] for cusip in returns_cusips}

        # Filter Compustat data to include only CUSIPs in returns data
        original_shape = self.data.shape
        self.data = self.data[self.data['cusip_match'].isin(returns_cusips_8digit)]

        # Remove the temporary matching column
        self.data = self.data.drop(columns=['cusip_match'])

        # Count unique CUSIPs in Compustat data after filtering
        compustat_cusips_after = self.data[self.id_column].nunique()

        # Report results
        print(f"Filtered Compustat data from {original_shape[0]:,} to {self.data.shape[0]:,} rows")
        print(f"Retained {compustat_cusips_after:,} unique CUSIPs out of {compustat_cusips_before:,} ({compustat_cusips_after/compustat_cusips_before*100:.1f}%)")
        print(f"Coverage: Found {compustat_cusips_after:,} out of {unique_returns_cusips:,} returns CUSIPs in Compustat data ({compustat_cusips_after/unique_returns_cusips*100:.1f}%)")

        return self





