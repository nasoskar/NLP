from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(df):

    X_train, X_temp, y_train, y_temp = train_test_split(df['text'], df['label'], train_size = 0.8, shuffle=True, random_state=42) 

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size = 0.5, shuffle=True, random_state=42) 

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
    

class FinBertDataset():
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class LSTMDataset():
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass