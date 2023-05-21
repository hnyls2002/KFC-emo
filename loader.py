import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def load_data(data_path="data/", emo_type="goemotion", extended_flag=False):
    print(">>>>>>>>>>>> Loading Data >>>>>>>>>>>>")
    # Check if processed CSV files exist
    if os.path.exists(data_path + "train.csv") and os.path.exists(data_path + "dev.csv") and os.path.exists(data_path + "test.csv"):
        # Load processed CSV files
        if extended_flag:
            train = pd.read_csv(data_path + "extended_train.csv")
            # # split extended set into 10 parts and write to disk
            # for i in range(10):
            #     train_part = train.iloc[i * 100000: (i + 1) * 100000]
            #     train_part.to_csv(
            #         data_path + "extended_train_" + str(i) + ".csv", index=False)
            # exit()
            # load extended train set from 10 parts
            # train = pd.DataFrame()
            # for i in range(8, 10):
            #     print(">>>>>>>>>>>> Loading extended train set part: ", i)
            #     train_part = pd.read_csv(
            #         data_path + "extended_train_" + str(i) + ".csv")
            #     train = pd.concat([train, train_part])
        else:
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

    print(">>>>>>>>>>>> The size of train/dev/test set is: ",
          train.shape, dev.shape, test.shape)

    # Read the emotions list
    with open(data_path + "emotions.txt", "r") as f:
        emotion_list = [line.strip() for line in f.readlines()]
    with open(data_path + "sentiments.txt", "r") as f:
        sentiment_list = [line.strip() for line in f.readlines()]
    with open(data_path + "ekmans.txt", "r") as f:
        ekman_list = [line.strip() for line in f.readlines()]

    # sentiment mapping
    json_file = open(data_path + "sentiment_mapping.json", 'r')
    sentiment_mapping = json.load(json_file)
    for dataset in (train, dev, test):
        for sentiment in sentiment_list:
            dataset[sentiment] = pd.Series([0] * len(dataset))
            for emo in sentiment_mapping[sentiment]:
                dataset[sentiment] |= dataset[emo]
        dataset["senti_sum"] = dataset[sentiment_list].apply(
            lambda x: sum([x[sentiment] for sentiment in sentiment_list]), axis=1)

    # ekman mapping
    json_file = open(data_path + "ekman_mapping.json", 'r')
    ekman_mapping = json.load(json_file)
    for dataset in (train, dev, test):
        for ekman in ekman_list:
            dataset[ekman] = pd.Series([0] * len(dataset))
            for emo in ekman_mapping[ekman]:
                dataset[ekman] |= dataset[emo]
        dataset["ekman_sum"] = dataset[ekman_list].apply(
            lambda x: sum([x[ekman] for ekman in ekman_list]), axis=1)

    # write first 100 line to test.txt
    # dataset.head(100).to_csv(data_path + "test.txt", index=False)

    # print(">>>>>>>>>>>> Extending Train Set >>>>>>>>>>>>")
    # extended_train = train
    # new_rows = []
    # while len(extended_train) + len(new_rows) < int(1e6):
    #     # randomly select two row
    #     print("extending train set: ", len(
    #         extended_train) + len(new_rows), " / ", int(1e6))

    #     idx1, idx2 = train.sample(2).index

    #     if idx1 != idx2:
    #         text = train.loc[idx1, "text"] + " " + train.loc[idx2, "text"]
    #         row = [0] * len(extended_train.columns)
    #         row[extended_train.columns.get_loc("text")] = text
    #         for emo in emotion_list:
    #             if train.loc[idx1, emo] or train.loc[idx2, emo]:
    #                 row[extended_train.columns.get_loc(emo)] = 1
    #         new_rows.append(row)

    # extended_train = pd.concat(
    #     [extended_train, pd.DataFrame(new_rows, columns=extended_train.columns)])

    # # store extended train set
    # extended_train.to_csv(data_path + "extended_train.csv", index=False)

    for dataset in (train, dev, test):
        if emo_type == "goemotion":
            dataset["label"] = dataset[emotion_list].apply(
                lambda x: torch.tensor([x[emotion] for emotion in emotion_list]), axis=1)
        elif emo_type == "sentiment":
            dataset["label"] = dataset[sentiment_list].apply(
                lambda x: torch.tensor([x[sentiment] for sentiment in sentiment_list]), axis=1)
        elif emo_type == "ekman":
            dataset["label"] = dataset[ekman_list].apply(
                lambda x: torch.tensor([x[ekman] for ekman in ekman_list]), axis=1)
        else:
            print("Error: Invalid emo_type")
            exit(1)

    # print example
    print("======================================example==========================================")
    print(train.head()[["text", "label"]])
    print("=======================================================================================")

    return train, dev, test
