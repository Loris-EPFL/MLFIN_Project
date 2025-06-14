# Machine Learning in Finance Project

## Overview

This repository contains our implementation of a machine learning-based trading strategy for a hedge fund. The project focuses on predicting monthly stock returns using financial data from Compustat and other predictors.

## Project Structure

```
├── datasets/                      # Processed datasets ready for modeling
├── preprocessing/                 # Data preprocessing scripts
├── .gitignore                     # Git ignore file
├── Machine_Learning_in_Finance___Project_Instructions.pdf  # Project instructions
├── README.md                      # This file
├── cleaner.ipynb                  # Notebook for cleaning data and handling NaN values
├── merge_compustat_returns.ipynb  # Notebook for merging Compustat and returns data
├── predictors_and_target.ipynb    # Notebook for preparing features and target variables in the sentiment analysis
├── preprocess_compustat_data.ipynb # Notebook for processing Compustat financial data
├── preprocess_returns.ipynb       # Notebook for processing returns data and adding MAs
├── sentiment_analysis.ipynb       # Notebook for sentiment analysis on financial texts
├── sentiment_analysis.py          # Python script for sentiment analysis
└── utils.py                       # Utility functions used across notebooks
```

## Data Sources

Our project uses several data sources:

1. **Monthly CRSP Returns**: Historical stock returns data
2. **Compustat Quarterly Data**: Fundamental financial indicators
3. **Sentiment Analysis**: Text-based sentiment from financial documents using 10k and Earning Calls

## Datasets

### Processed Datasets
Download our processed datasets ready for modeling:
[Download Processed Datasets](https://drive.google.com/file/d/17eu2IgeY3t8CmOypehcUbf0L9oDqfFvf/view?usp=sharing)

### Raw Datasets
Access the original raw datasets:
[Download Raw Datasets](https://drive.google.com/drive/folders/1TGSfQOj_laBfIwR4yCVWCKRdq2_0lZLE)

## Data Preprocessing

Our preprocessing includes:

1. **Returns Data Processing**:
   - Handling missing values
   - Calculating moving averages (SMA and EMA, for 3, 6, and 12 months windows)

2. **Compustat Data Processing**:
   - Removing NaNs and Outliers
   - Selecting high-coverage financial indicators (>65% data availability)
   - Computing financial ratios
   - Normalizing financial metrics

4. **Sentiment Analysis**:
   - Extracting sentiment from financial texts
   - Generating sentiment scores
   - Integrating sentiment with quantitative data

5. **Data Merging**:
   - Aligning time series data using `pd.merge_asof`
   - Matching companies across different datasets using their CUSIP or gvkey
   - Creating a unified dataset for modeling

## Feature Engineering

We engineered several features from the raw data:

1. **Financial Ratios**:
   - Earnings Growth
   - Revenue Growth
   - CAPEX Growth

2. **Returns Indicators**:
   - Moving Average (Simple and Exponential) (3, 6, 12-month)

3. **Sentiment Features**:
   - Sentiment polarity scores
   - Topic modeling results
   - Text-based risk indicators

## Modeling Approach

add model desc

## Usage

### Data Preparation

To prepare the data, run the following notebooks in sequence:
1. [preprocess_returns.ipynb](preprocess_returns.ipynb)
2. [preprocess_compustat_data.ipynb](preprocess_compustat_data.ipynb)
3. [sentiment_analysis.ipynb](sentiment_analysis.ipynb)
4. [merge_compustat_returns.ipynb](merge_compustat_returns.ipynb)
5. [cleaner.ipynb](cleaner.ipynb)
6. [predictors_and_target.ipynb](predictors_and_target.ipynb)

### Model Training

```python
# Train the model
python train_model.py --model deep_factor --epochs 100 --batch_size 32
```

### Backtesting

```python
# Run backtest
python backtest.py --model_path models/deep_factor.h5 --start_date 2010-01-01 --end_date 2020-12-31
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tensorflow 2.x
- pytorch
- matplotlib
- seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Results

Our model achieves:
- Sharpe ratio: [TBD]
- Annual return: [TBD]
- Maximum drawdown: [TBD]

## Contributors

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]
- [Team Member 5]

