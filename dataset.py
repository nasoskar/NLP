from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import re
from config import *
import torch

def split_dataset(df):

    X_train, X_temp, y_train, y_temp = train_test_split(df['text'], df['label'], train_size = 0.8, shuffle=True, stratify=df['label'], random_state=42) 

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size = 0.5, shuffle=True, stratify=y_temp, random_state=42) 

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataframe():
    phrasebank_df = pd.read_csv(
        'FinancialPhraseBank-v1.0/Sentences_75Agree.txt',
        sep='@',
        header=None,
        names=['text', 'sentiment'],
        encoding='latin-1'  # needed for special characters
    )

    label_map = {"neutral": 2, "positive": 1, "negative": 0}
    phrasebank_df['sentiment'] = phrasebank_df['sentiment'].map(label_map)

    phrasebank_df.rename(columns={"sentiment": "label"}, inplace=True)

    splits = {'train': 'sent_train.csv', 'validation': 'sent_valid.csv'}
    df_tr = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["train"])
    df_val = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["validation"])
    twitter_df = pd.concat([df_tr, df_val], ignore_index=True)

    total_df = pd.concat([phrasebank_df, twitter_df], ignore_index=True, sort=False)

    return total_df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'\$[a-zA-Z]+', 'TICKER', text) # normalize tickers
    return text

def build_vocab(texts, min_freq=1):
    from collections import Counter
    
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)
    
    # 0 = <PAD>, 1 = <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab
    

class FinBertDataset():
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class LSTMClassification(Dataset):
    def __init__(self, X_train, y_train, vocab):
        self.data = X_train.reset_index(drop=True)
        self.labels = y_train.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx].split() #split the sentence into tokens
        ids = [self.vocab.get(token, 1) for token in tokens]

        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        elif len(ids) < MAX_LEN:
            ids += [0] * (MAX_LEN - len(ids))

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        