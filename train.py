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

def LSTM_train(model, optimizer, criterion, train_dataloader, val_dataloader):
    best_val_loss = float('inf')
    patience_counter = 0 #used for early stopping
    min_delta = 0.005 # Minimum change in loss to qualify as an improvement
    best_val_f1 = 0

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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

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

    return best_val_f1

def finbert_train(model, optimizer, criterion, train_dataloader, val_dataloader):
    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0
    min_delta = 0.005

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.

        for data in train_dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["label"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.
        all_preds = []
        all_labels= []

        with torch.no_grad():
            for data in val_dataloader:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                labels = data["label"].to(device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")

        if val_loss < best_val_loss - min_delta:
            patience_counter = 0
            best_val_loss =  val_loss
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/best_finbert.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_dataloader):.4f} | Val Loss: {val_loss/len(val_dataloader):.4f} | Val Macro F1: {val_f1:.4f}")

    return best_val_f1


