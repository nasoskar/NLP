
Financial Sentiment Analysis —  README file
Author: Nasos Karas | 250044150 | MSc Artificial Intelligence
Course: Natural Language Processing


### PACKAGE CONTENTS


    inference.ipynb         - Main notebook to run inference on test set using two best trained models
    main.ipynb              - The main notebook to run all models, generate plots and execute the workflow end-to-end
    config.py               - Hyperparameters and constants
    dataset.py              - Data loading and preprocessing
    svm.py                  - SVM model class
    finbert.py              - FinBERT model class
    requirements.txt        - Required Python libraries
    README.md               - This file
    lstm.py                 - LSTM model class
    test.csv                - A csv file that contains only the test set of the dataset
    train.py                - Loads GloVe word embeddings and calls the functions that train LSTM and FinBERT

    checkpoints/
        best_svm.pkl        - Saved SVM model (vectorizer + classifier)
        best_finbert.pth    - Saved FinBERT weights


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


### REQUIREMENTS

To run all necessary files, execute pip install -r requirements.txt, to install all necessary dependencies

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

