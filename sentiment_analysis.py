import pandas as pd
import os
import re
from tqdm import tqdm
import torch
from transformers import pipeline

class ReportsProcessing:
    def __init__(self, data_path: str, clean_folder: str):
        self.data = self._load_clean_data(data_path)
        self.clean_folder = clean_folder
        os.makedirs(self.clean_folder, exist_ok=True)
        self.device = 'cuda'  # Set to 'cuda' if needed

    @staticmethod
    def _load_clean_data(file_path: str) -> pd.DataFrame:
        data = pd.read_parquet(file_path)
        data['report_date'] = pd.to_datetime(data['report_date'])
        return data

    def clean_reports(self):
        tqdm.pandas(desc="Cleaning Reports", unit='reports')
        self.data['cleaned_text'] = self.data['text'].progress_apply(self._clean_text)
        self.data.drop(columns=['text'], inplace=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"^(item 7\.)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r'[\n\r\t]+', " ", text)
        text = re.sub(r"\s+", " ", text)
        cleaned = text.strip().lower()
        return cleaned[:512]

    def reports_sentiment_analysis(self, batch_size: int = 128):
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if self.device == 'cuda' else -1
        )

        texts = self.data['cleaned_text'].tolist()
        sentiments = []
        sentiment_scores = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Detecting sentiments"):
            batch = texts[i:i + batch_size]
            results = sentiment_classifier(batch, batch_size=batch_size)

            sentiments.extend([r['label'] for r in results])
            sentiment_scores.extend([r['score'] for r in results])

        self.data['sentiment'] = sentiments
        self.data['sentiment_score'] = sentiment_scores

        sentiment_summary = self.data[['sentiment', 'sentiment_score', 'cleaned_text']].groupby('sentiment').agg({
            'cleaned_text': 'count',
            'sentiment_score': 'mean'
        }).rename(columns={'cleaned_text': 'report_count', 'sentiment_score': 'avg_sentiment_score'})

        print("Sentiment Analysis Report:")
        for label in ['positive', 'neutral', 'negative']:
            if label in sentiment_summary.index:
                count = sentiment_summary.loc[label, 'report_count']
                avg_score = sentiment_summary.loc[label, 'avg_sentiment_score']
                print(f"\t- {label.title()}: {count} tweets (Avg. score: {avg_score:.2f})")
            else:
                print(f"\t- {label.title()}: 0 tweets (Avg. score: N/A)")

    def save_clean_data(self):
        self.data.to_parquet(
            os.path.join(self.clean_folder, 'sentiment_tenkreports.parquet'), index=False
        )

