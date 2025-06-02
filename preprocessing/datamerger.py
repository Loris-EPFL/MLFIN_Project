import pandas as pd
import numpy as np
from pathlib import Path
import gc  # Garbage collector

class DataMerger:
    """
    A class for efficiently merging large financial datasets using pd.merge_asof.
    """
    
    def __init__(self, date_column='datadate'):
        """
        Initialize the DataMerger.
        
        Parameters:
        -----------
        date_column : str
            Column name for the date to merge on
        """
        self.date_column = date_column
    
    def convert_to_parquet(self, csv_path, parquet_path=None, chunksize=500000, sort=True):
        """
        Convert a large CSV file to Parquet format, optionally sorting by date.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        parquet_path : str, optional
            Path to save the Parquet file (if None, uses CSV path with .parquet extension)
        chunksize : int
            Number of rows to process at a time
        sort : bool
            Whether to sort the data by date before saving
            
        Returns:
        --------
        str
            Path to the created Parquet file
        """
        if parquet_path is None:
            parquet_path = str(Path(csv_path).with_suffix('.parquet'))
        
        print(f"Converting {csv_path} to Parquet format...")
        
        # Create directory if it doesn't exist
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Process in chunks to avoid memory issues
        chunk_list = []
        total_rows = 0
        
        for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            print(f"Processing chunk {chunk_num+1} with {len(chunk)} rows")
            
            # Convert date column to datetime
            if self.date_column in chunk.columns:
                chunk[self.date_column] = pd.to_datetime(chunk[self.date_column])
            
            # Collect chunks for sorting later
            chunk_list.append(chunk)
            total_rows += len(chunk)
            
            # Free memory
            gc.collect()
        
        # Combine all chunks
        print("Combining chunks...")
        combined_data = pd.concat(chunk_list, ignore_index=True)
        
        # Clear chunk list to free memory
        chunk_list = []
        gc.collect()
        
        # Sort by date if requested
        if sort and self.date_column in combined_data.columns:
            print(f"Sorting data by {self.date_column}...")
            combined_data = combined_data.sort_values(self.date_column)
        
        # Save to Parquet
        print(f"Saving {total_rows} rows to {parquet_path}")
        combined_data.to_parquet(parquet_path, index=False)
        
        # Free memory
        del combined_data
        gc.collect()
        
        return parquet_path
    
    def merge_files_preserve_all(self, file_paths, merge_on=None, by=None, direction='forward', 
                               tolerance=None, output_path=None, chunksize=500000):
        """
        Merge multiple large files using pd.merge_asof, preserving all rows.
        
        Parameters:
        -----------
        file_paths : list
            List of paths to the files to merge (CSV or Parquet)
        merge_on : str, optional
            Column to merge on (if None, uses self.date_column)
        by : str or list, optional
            Columns to match exactly (e.g., company identifier)
        direction : str
            Direction of merge ('backward', 'forward', or 'nearest')
        tolerance : pd.Timedelta, optional
            Maximum time difference for matching
        output_path : str, optional
            Path to save the merged data
        chunksize : int
            Number of rows to process at a time
            
        Returns:
        --------
        pandas.DataFrame or str
            The merged data or path to the saved file
        """
        if merge_on is None:
            merge_on = self.date_column
        
        # Convert all files to Parquet if they're CSV
        parquet_paths = []
        for file_path in file_paths:
            if file_path.endswith('.csv'):
                parquet_path = self.convert_to_parquet(file_path, sort=True, chunksize=chunksize)
                parquet_paths.append(parquet_path)
            else:
                parquet_paths.append(file_path)
        
        print(f"Merging {len(parquet_paths)} files...")
        
        # First, concatenate all files to ensure we don't lose any rows
        print("First concatenating all files to preserve all rows...")
        all_dfs = []
        
        for i, parquet_path in enumerate(parquet_paths):
            print(f"Reading file {i+1}/{len(parquet_paths)}: {parquet_path}")
            
            # Read the file
            df = pd.read_parquet(parquet_path)
            
            # Ensure the merge column is datetime if it's a date
            if pd.api.types.is_datetime64_dtype(df[merge_on]):
                df[merge_on] = pd.to_datetime(df[merge_on])
            
            # Add to list
            all_dfs.append(df)
            
            # Free memory
            gc.collect()
        
        # Concatenate all dataframes
        print("Combining all files...")
        combined_data = pd.concat(all_dfs, ignore_index=True)
        
        # Free memory
        all_dfs = []
        gc.collect()
        
        # Sort by date and company identifier
        print(f"Sorting by {merge_on} and {by}...")
        if isinstance(by, list):
            sort_cols = [merge_on] + by
        else:
            sort_cols = [merge_on, by]
        
        combined_data = combined_data.sort_values(sort_cols)
        
        # Remove duplicate rows if they exist (same date and company)
        print("Removing any duplicate rows...")
        if isinstance(by, list):
            dedup_cols = [merge_on] + by
        else:
            dedup_cols = [merge_on, by]
        
        # Count before deduplication
        count_before = len(combined_data)
        
        # Drop duplicates
        combined_data = combined_data.drop_duplicates(subset=dedup_cols, keep='first')
        
        # Count after deduplication
        count_after = len(combined_data)
        print(f"Removed {count_before - count_after} duplicate rows")
        
        # Save the result if requested
        if output_path:
            print(f"Saving merged data to {output_path}")
            
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.endswith('.csv'):
                combined_data.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                combined_data.to_parquet(output_path, index=False)
            else:
                # Default to Parquet
                output_path = output_path + '.parquet'
                combined_data.to_parquet(output_path, index=False)
            
            return output_path
        
        return combined_data
    
    def merge_files_with_asof(self, file_paths, merge_on=None, by=None, direction='forward', 
                            tolerance=None, output_path=None, chunksize=500000):
        """
        Merge multiple large files using pd.merge_asof, ensuring all rows are preserved.
        
        Parameters:
        -----------
        file_paths : list
            List of paths to the files to merge (CSV or Parquet)
        merge_on : str, optional
            Column to merge on (if None, uses self.date_column)
        by : str or list, optional
            Columns to match exactly (e.g., company identifier)
        direction : str
            Direction of merge ('backward', 'forward', or 'nearest')
        tolerance : pd.Timedelta, optional
            Maximum time difference for matching
        output_path : str, optional
            Path to save the merged data
        chunksize : int
            Number of rows to process at a time
            
        Returns:
        --------
        pandas.DataFrame or str
            The merged data or path to the saved file
        """
        if merge_on is None:
            merge_on = self.date_column
        
        # Convert all files to Parquet if they're CSV
        parquet_paths = []
        for file_path in file_paths:
            if file_path.endswith('.csv'):
                parquet_path = self.convert_to_parquet(file_path, sort=True, chunksize=chunksize)
                parquet_paths.append(parquet_path)
            else:
                parquet_paths.append(file_path)
        
        print(f"Merging {len(parquet_paths)} files...")
        
        # Load all dataframes first to analyze columns
        all_dfs = []
        for i, path in enumerate(parquet_paths):
            print(f"Loading file {i+1}/{len(parquet_paths)}: {path}")
            df = pd.read_parquet(path)
            
            # Ensure the merge column is datetime if it's a date
            if pd.api.types.is_datetime64_dtype(df[merge_on]):
                df[merge_on] = pd.to_datetime(df[merge_on])
            
            # Sort by the merge column
            df = df.sort_values(merge_on)
            
            all_dfs.append(df)
        
        # Identify unique columns across all dataframes
        all_columns = set()
        for df in all_dfs:
            all_columns.update(df.columns)
        
        # Create a base dataframe with all unique dates and companies
        print("Creating a base dataframe with all unique dates and companies...")
        
        # Extract all unique dates and companies
        all_dates = set()
        all_companies = set()
        
        for df in all_dfs:
            all_dates.update(df[merge_on].tolist())
            if isinstance(by, list):
                for col in by:
                    all_companies.update(df[col].tolist())
            else:
                all_companies.update(df[by].tolist())
        
        # Create a cross product of all dates and companies
        print(f"Creating cross product of {len(all_dates)} dates and {len(all_companies)} companies...")
        
        # This could be memory intensive for large datasets
        dates_df = pd.DataFrame({merge_on: list(all_dates)})
        
        if isinstance(by, list):
            # Handle multiple 'by' columns
            # This is more complex and would need a custom approach
            # For simplicity, let's just use the first 'by' column for now
            companies_df = pd.DataFrame({by[0]: list(all_companies)})
            base_df = dates_df.assign(key=1).merge(companies_df.assign(key=1), on='key').drop('key', axis=1)
        else:
            companies_df = pd.DataFrame({by: list(all_companies)})
            base_df = dates_df.assign(key=1).merge(companies_df.assign(key=1), on='key').drop('key', axis=1)
        
        print(f"Base dataframe has {len(base_df)} rows")
        
        # Sort the base dataframe
        base_df = base_df.sort_values(merge_on)
        
        # Now merge each dataframe with the base
        merged_df = base_df.copy()
        
        for i, df in enumerate(all_dfs):
            print(f"Merging dataframe {i+1}/{len(all_dfs)} with base...")
            
            # Get columns that are unique to this dataframe
            df_columns = set(df.columns) - {merge_on, by} if isinstance(by, str) else set(df.columns) - {merge_on}.union(by)
            
            # Perform the asof merge
            merged_df = pd.merge_asof(
                merged_df, 
                df[list({merge_on, by} if isinstance(by, str) else {merge_on}.union(by)) + list(df_columns)],
                on=merge_on,
                by=by,
                direction=direction,
                tolerance=tolerance
            )
            
            print(f"Merged result: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            
            # Free memory
            gc.collect()
        
        # Save the result if requested
        if output_path:
            print(f"Saving merged data to {output_path}")
            
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.endswith('.csv'):
                merged_df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                merged_df.to_parquet(output_path, index=False)
            else:
                # Default to Parquet
                output_path = output_path + '.parquet'
                merged_df.to_parquet(output_path, index=False)
            
            return output_path
        
        return merged_df
