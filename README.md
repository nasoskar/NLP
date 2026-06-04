# Does Context Always Help?
### A Length-Stratified Comparison of SVM, LSTM, and FinBERT for Financial Sentiment Classification

**Author:** Nasos Karas | 250044150  
**Course:** MSc Artificial Intelligence — Natural Language Processing  
**Institution:** City, University of London  

---

## Overview

This project investigates whether contextual representation models consistently outperform non-contextual models in financial sentiment analysis, and whether this advantage varies by sentence length. Three models of increasing contextual sophistication are compared:

| Model | Type | Description |
|-------|------|-------------|
| SVM + TF-IDF | Non-contextual baseline | Bag-of-words with unigrams and bigrams |
| LSTM + GloVe | Sequential neural | GloVe 100d embeddings, left-to-right processing |
| FinBERT | Contextual transformer | Pretrained on financial corpora, bidirectional attention |

All models are evaluated overall and stratified by sentence length (short ≤12 words, long >12 words).

---

## Key Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| SVM + TF-IDF | 0.834 | 0.749 |
| LSTM + GloVe | 0.782 | 0.729 |
| **FinBERT** | **0.843** | **0.872** |

| Model | Short (≤12) (Macro F1) | Long (>12) (Macro F1) |
|-------|-------------|------------|
| SVM + TF-IDF | 0.790 | 0.741 |
| LSTM + GloVe | 0.743 | 0.714 |
| **FinBERT** | **0.812** | **0.828** |


**Main finding:** FinBERT's advantage over SVM grows from 0.022 on short texts to 0.087 on long texts, confirming that contextual representations become more valuable as text length increases. Surprisingly, SVM outperforms LSTM on all metrics. For short texts under computational constraints, SVM with TF-IDF emerges as the most practical choice — FinBERT's advantage diminishes significantly while SVM trains in minutes on CPU compared to FinBERT's 2+ hours on GPU, making it the best trade-off between performance and efficiency for real-time applications such as financial tweet analysis.

---

## Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| Twitter Financial News Sentiment | 11,932 | [HuggingFace](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| Financial PhraseBank (75% agree) | 3,453 | [HuggingFace](https://huggingface.co/datasets/takala/financial_phrasebank) |
| **Combined** | **15,385** | Merged and split 80/10/10 |

---

## Project Structure

```
financial-sentiment-length-study/
├── config.py               # Hyperparameters and constants
├── dataset.py              # Data loading, preprocessing, Dataset classes
├── svm.py                  # SVM + TF-IDF classifier
├── lstm.py                 # LSTM nn.Module definition
├── finbert.py              # FinBERT + classification head
├── train.py                # Training loops (LSTM, FinBERT) + GloVe loader
├── evaluate.py             # Evaluation metrics and plots
├── main.ipynb              # Full training and evaluation pipeline
├── inference.ipynb         # Load best models and run on test set
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Setup

### Requirements

- Python 3.12
- Google Colab (recommended, T4 GPU) or local machine
- GloVe embeddings: [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) — extract `glove.6B.100d.txt`
- Financial PhraseBank: download `Sentences_75Agree.txt` from [here](https://huggingface.co/datasets/takala/financial_phrasebank)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Files (place in Google Drive at `/MyDrive/NLP/`)

```
NLP/
├── glove.6B.100d.txt
├── FinancialPhraseBank-v1.0/
│   └── Sentences_75Agree.txt
└── checkpoints/
    ├── best_svm.pkl
    └── best_finbert.pth
```

---

## Running the Project

### Option A — Google Colab (Recommended)

1. Open `main.ipynb` in Google Colab
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run Cell 1 to mount Google Drive
4. Run Cell 2 to clone the repo and install dependencies
5. Run all cells top to bottom

### Option B — Local Machine

```bash
# Clone the repo
git clone [link]
cd financial-sentiment-length-study

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook main.ipynb
```

---



## Preprocessing

All models use the same preprocessing pipeline:

Length buckets:
- **Short:** ≤ 12 words
- **Long:** > 12 words

Class imbalance handled via `class_weight='balanced'` (SVM) and `CrossEntropyLoss(weight=class_weights)` (LSTM, FinBERT).

---

## Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.3.0 | LSTM and FinBERT training |
| `transformers` | 4.40.0 | FinBERT model and tokenizer |
| `scikit-learn` | 1.4.2 | SVM, TF-IDF, GridSearchCV, metrics |
| `pandas` | 2.2.2 | Data loading and preprocessing |
| `numpy` | 1.26.4 | Numerical operations |
| `matplotlib` | 3.9.0 | Plots and visualisations |
| `seaborn` | 0.13.2 | Heatmaps |
| `datasets` | 2.20.0 | Loading Twitter dataset |



### TEST SET

The test set (test.csv) is available at the following shared Google Drive link:

    https://drive.google.com/file/d/19aya_AmttLWbX9fCOKjNkYW0wV73sfeI/view?usp=drive_link


Download test.csv and upload it to your Google Drive at:
    /content/drive/MyDrive/NLP/test.csv

Alternatively the full dataset can be obtained from:
    - Twitter Financial News Sentiment:
      https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
    - Financial PhraseBank:
      https://huggingface.co/datasets/takala/financial_phrasebank


### HOW TO RUN IN GOOGLE COLAB 

STEP 1: Open Google Colab
    - Go to https://colab.research.google.com
    - Upload inference.ipynb

STEP 2: Set Runtime to GPU (optional but faster)
    - Runtime → Change runtime type → T4 GPU → Save

STEP 3: Mount Google Drive
    - Run respective Cell in the notebook
    - Sign in to your Google account when prompted

STEP 4: Upload Checkpoints to Drive
    - Upload the checkpoints/ folder to your Google Drive at:
      /content/drive/MyDrive/NLP/checkpoints/
    - The folder can be found at https://drive.google.com/drive/folders/1R6jITjTAUOLFEaRdpjzCzRGaqtyoVj9E?usp=sharing

STEP 5: Upload Test Set to Drive
    - Upload test.csv to your Google Drive at:
      /content/drive/MyDrive/NLP/test.csv
    - OR download from the link provided above

STEP 6: Run All Cells
    - Runtime → Run all
    - The notebook will:
        1. Load and preprocess the test set
        2. Run SVM inference and print results
        3. Run FinBERT inference and print results
        4. Display confusion matrices for both models
        5. Print final comparison table

### NOTES

- GloVe embeddings are only required for LSTM (not included in this package
  as the two best models are SVM and FinBERT)
- If running on CPU, FinBERT inference may take 5-10 minutes
- If running on T4 GPU, FinBERT inference takes under 1 minute
- The Drive mount cell will always ask for authentication with your own
  Google account

