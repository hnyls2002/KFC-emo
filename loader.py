import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

def load_data(data_path="data/"):
    print(">>>>>>>>>>>> Loading Data >>>>>>>>>>>>")
    # Check if processed CSV files exist
    if os.path.exists(data_path + "train.csv") and os.path.exists(data_path + "dev.csv") and os.path.exists(data_path + "test.csv"):
        # Load processed CSV files
        train = pd.read_csv(data_path + "train.csv")
        dev = pd.read_csv(data_path + "dev.csv")
        test = pd.read_csv(data_path + "test.csv")
    else:
        # Load raw CSV files
        full_data_path = data_path + "full_dataset/"
        df1 = pd.read_csv(full_data_path + "goemotions_1.csv")
        df2 = pd.read_csv(full_data_path + "goemotions_2.csv")
        df3 = pd.read_csv(full_data_path + "goemotions_3.csv")
        df = pd.concat([df1, df2, df3])

        # Split data into train/dev/test sets
        train, test = train_test_split(df, test_size=0.2)
        train, dev = train_test_split(train, test_size=0.25)

        # Save data to disk
        train.to_csv(data_path + "train.csv", index=False)
        dev.to_csv(data_path + "dev.csv", index=False)
        test.to_csv(data_path + "test.csv", index=False)

    # Read the emotions list
    with open(data_path + "emotions.txt", "r") as f:
        emotion_list = [line.strip() for line in f.readlines()]
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_list)}

    for dataset in (train,dev,test):
        # Add emotion column
        dataset["emotion"] = dataset[emotion_list].apply(lambda x: x.idxmax(), axis=1)
        # Add label which is 28-length one-hot vector
        dataset["label"] = dataset["emotion"].apply(lambda x: F.one_hot(torch.tensor(emotion_to_idx[x]), num_classes=len(emotion_list)))

    # print example
    print("======================================example==========================================")
    print(train.head()[["text", "emotion", "label"]])
    print("=======================================================================================")

    return train, dev, test

