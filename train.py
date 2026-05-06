from dataset import create_dataframe, split_dataset, clean_text, LSTMClassification, build_vocab
from svm import SVM_Classifier
from torch.utils.data import DataLoader
from config import *
from lstm import *
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_glove(glove_path, vocab, embedding_dim=100):
    # initialize with zeros
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            if word in vocab:
                embedding_matrix[vocab[word]] = vector
    
    return torch.tensor(embedding_matrix, dtype=torch.float)

def LSTM_train(model, optimizer, criterion):
    best_val_loss = float('inf')
    patience_counter = 0 #used for early stopping
    min_delta = 0.005 # Minimum change in loss to qualify as an improvement
    

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.

        for data in train_dataloader:

            inputs = data["input_ids"].to(device)
            labels = data["label"].to(device)

            optimizer.zero_grad() #zero the grads in each epoch
            output = model(inputs)

            #compute the loss
            loss = criterion(output, labels)  
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.

        all_preds = []
        all_labels = []

        with torch.no_grad():  
            for data in val_dataloader:
                inputs = data["input_ids"].to(device)
                labels = data["label"].to(device)
                
                output = model(inputs)
                loss = criterion(output, labels) 
                #loss   = loss_function(output, labels)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")

        if val_loss < best_val_loss - min_delta:
            patience_counter = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_lstm.pth")
        else:
            patience_counter += 1       
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_dataloader):.4f} | Val Loss: {val_loss/len(val_dataloader):.4f} | Val Macro F1: {val_f1:.4f}")

if __name__ == "__main__":

    #---PROCESS---
    #Step 1: create dataframe based on the image paths
    df = create_dataframe()
    df['text'] = df['text'].apply(clean_text)
    print(df)

    #Step 2: Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)

    #Step 3: SVM with TF-IDF
    svmclass = SVM_Classifier()
    svmclass.fit(X_train, y_train)
    y_pred = svmclass.predict(X_test)


    vocab = build_vocab(X_train)

    train_dataset = LSTMClassification(X_train, y_train, vocab)
    val_dataset   = LSTMClassification(X_val,   y_val,   vocab)  
    test_dataset  = LSTMClassification(X_test,  y_test,  vocab)  

    #Step 4:  Call DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    batch = next(iter(train_dataloader))
    input_ids = batch["input_ids"]  # shape: (32, 30)
    labels    = batch["label"]      # shape: (32,)

    # Step 5: Call LSTM model
    glove_matrix = load_glove("glove.6B.100d.txt", vocab)
    weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    model = LSTM(len(vocab), 100, HIDDEN_SIZE, 3).to(device)
    model.embedding.weight.data.copy_(glove_matrix)
    model.embedding.weight.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    LSTM_train(model, optimizer, criterion)


