"""
This python script is used to define functions for transforming and aggregating earnings calls and 10-K reports sentiment results with other predictors.
"""
import pandas as pd

# Prepare earnings calls sentiment results
def transform_earnings_calls(df):
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    df['mostimportantdateutc'] = pd.to_datetime(df['mostimportantdateutc'])
    df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('int')
    df['weighted_sentiment'] = df['sentiment_label'] * df['sentiment_score']
    return df

# Prepare 10-K sentiment results
def transform_tenkreports(df):
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['cik'] = pd.to_numeric(df['cik'], errors='coerce').astype('int')
    df['weighted_sentiment'] = df['sentiment_label'] * df['sentiment_score']
    return df

# Aggregate earnings calls sentiment results
def aggregate_earnings_calls(df):
    return df.groupby(['gvkey','mostimportantdateutc'], as_index=False).agg(
    weighted_sentiment_earnings_calls=('weighted_sentiment', 'mean'))

# Aggregate 10-K sentiment results
def aggregate_tenkreports(df):
    return df.groupby(['cik','report_date'], as_index=False).agg(
    weighted_sentiment_10k=('weighted_sentiment', 'mean'))

# Prepare dataframes for merging with earnings calls
def prepare_for_merge_ec(target, transcript_sentiment):
    target = target.sort_values(['MthCalDt']).reset_index(drop=True)
    transcript_sentiment = transcript_sentiment.sort_values(['mostimportantdateutc']).reset_index(drop=True)
    target['MthCalDt'] = target['MthCalDt'].astype('datetime64[ns]')
    transcript_sentiment['mostimportantdateutc'] = transcript_sentiment['mostimportantdateutc'].astype('datetime64[ns]')
    target['gvkey'] = target['gvkey'].astype(int)
    return target, transcript_sentiment

# Prepare dataframes for merging with 10-K reports
def prepare_for_merge_10k(target, transcript_sentiment):
    target = target.sort_values(['MthCalDt']).reset_index(drop=True)
    transcript_sentiment = transcript_sentiment.sort_values(['report_date']).reset_index(drop=True)
    target['MthCalDt'] = target['MthCalDt'].astype('datetime64[ns]')
    transcript_sentiment['report_date'] = transcript_sentiment['report_date'].astype('datetime64[ns]')
    target['cik'] = target['cik'].astype(int)
    return target, transcript_sentiment

# Fill missing values in earnings calls sentiment results
def fill_missing_earning_calls(df):
    df['mostimportantdateutc'] = df['mostimportantdateutc'].fillna(df['MthCalDt'])
    df['weighted_sentiment_earnings_calls'] = df['weighted_sentiment_earnings_calls'].fillna(0)
    return df

# Fill missing values in 10-K reports sentiment results
def fill_missing_10k(df):
    df['report_date'] = df['report_date'].fillna(df['MthCalDt'])
    df['weighted_sentiment_10k'] = df['weighted_sentiment_10k'].fillna(0)
    return df