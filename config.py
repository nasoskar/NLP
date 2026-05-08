################# LSTM ##################
MAX_LEN = 30
EPOCHS = 20
PATIENCE = 5
HIDDEN_SIZES = [64, 128, 256]
DROPOUT_RATES = [0, 0.3, 0.5]
LEARNING_RATES = [1e-3, 1e-4]
LSTM_BATCH_SIZE = 32

############## FINBERT ################
FINBERT_DROPOUTS = [0.1, 0.3]
FINBERT_LRS = [2e-5, 3e-5]  # standard range for fine-tuning transformers
FINBERT_BATCH_SIZE = 16