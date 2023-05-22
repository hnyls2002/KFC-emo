from flask import Flask, render_template, request
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from model import BertSentimentAnalysis

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir="./cache/", num_labels=28)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="./cache/")

with open("./data/emotions.txt", "r") as f:
    emo_list = [line.strip() for line in f.readlines()]

thresholds = pd.read_csv("./saving/demo/thresholds.csv", index_col=0)

model = BertSentimentAnalysis(bert_model, bert_model.config.hidden_size, 0.1, len(emo_list), dense_num=1)

# run on cpu only 
model.load_state_dict(torch.load("./saving/demo/parameters.pth", map_location=torch.device('cpu')))

emoji_dict = {}
with open("./saving/demo/emoji_dict.txt", "r") as f:
    for line in f.readlines():
        emo, emoji = line.strip().split(" ")
        emoji_dict[emo] = emoji

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST" :
        input_text = request.form.get("input_text")

        encoding = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        logits = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        logits = logits.squeeze()
        logits = torch.sigmoid(logits)
        predictions = []
        for i in range(len(emo_list)):
            if logits[i] > thresholds.loc[emo_list[i], "threshold"]:
                emotions = emoji_dict[emo_list[i]] + " : " + emo_list[i]
                predictions.append(emotions)
        return render_template("index.html", predictions=predictions)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)