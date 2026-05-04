from dataset import create_dataframe, split_dataset
if __name__ == "__main__":

    #---PROCESS---
    #Step 1: create dataframe based on the image paths
    df = create_dataframe()
    print(df)

    #Step 2: Split the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df)