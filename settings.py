import json
import os
import pandas as pd

saving_dir = "./saving/"


def try_load_hypara(exp_name):
    path = saving_dir + exp_name + '/' + "hypara.json"
    if os.path.exists(path):
        json_file = open(path, 'r')
        hypara = json.load(json_file)
        return hypara
    else:
        return None


def load_res(exp_name, emo_list):
    path = saving_dir + exp_name + '/' + "res.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return pd.DataFrame() 


def store_train(exp_name, idx, df):
    path = saving_dir + exp_name + '/' + "train" + str(idx) + ".csv"
    df.to_csv(path, index=False)


def store_res(exp_name, df):
    path = saving_dir + exp_name + '/' + "res.csv"
    df.to_csv(path, index=False)
