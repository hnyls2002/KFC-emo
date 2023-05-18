import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


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

        print(">>>>>>>>>>>> Splitting Data >>>>>>>>>>>>")

        # Split data into train/dev/test sets
        train, test = train_test_split(df, test_size=0.05)
        train, dev = train_test_split(train, test_size=1/19)

        new_train = []
        new_dev = []
        new_test = []

        for _, row in train.iterrows():
            if row["example_very_unclear"] == False:
                new_train.append(row)
        for _, row in dev.iterrows():
            if row["example_very_unclear"] == False:
                new_dev.append(row)
        for _, row in test.iterrows():
            if row["example_very_unclear"] == False:
                new_test.append(row)
        train = pd.DataFrame(new_train)
        dev = pd.DataFrame(new_dev)
        test = pd.DataFrame(new_test)

        # Save data to disk
        train.to_csv(data_path + "train.csv", index=False)
        dev.to_csv(data_path + "dev.csv", index=False)
        test.to_csv(data_path + "test.csv", index=False)

    print(">>>>>>>>>>>> The size of train/dev/test set is: ", train.shape, dev.shape, test.shape)

    # Read the emotions list
    with open(data_path + "emotions.txt", "r") as f:
        emotion_list = [line.strip() for line in f.readlines()]

    for dataset in (train, dev, test):
        dataset["label"] = dataset[emotion_list].apply(
            lambda x: torch.tensor([x[emotion] for emotion in emotion_list]), axis=1)
        dataset["label_sum"] = dataset["label"].apply(
            lambda x: torch.sum(x))
        # for _, row in dataset.iterrows():
        #     for idx, emo in enumerate(emotion_list):
        #         if row[emo] != row["label"][idx]:
        #             print(
        #                 "\x1b[31m" + ">>>>>>>>>>>> Warning: Inconsistent label detected in the following example:" + "\x1b[0m")
        #             print(row["text"])
        #     if row["label_sum"] == 0:
        #         print(
        #             "\x1b[31m" + ">>>>>>>>>>>> Warning: No emotion detected in the following example:" + "\x1b[0m")
        #         print(row["text"])

    # print example
    print("======================================example==========================================")
    print(train.head()[["text", "label_sum", "label"]])
    print("=======================================================================================")

    return train, dev, test
