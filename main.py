import os
import torch
from model import BertSentimentAnalysis
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loader import load_data
from bertpre import bert_init

# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
with open("./data/emotions.txt", "r") as f:
    emotion_list = [line.strip() for line in f.readlines()]
emotion_num = len(emotion_list)
train, dev, test = load_data()

# Set hyperparameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-5
dropout_prob = 0.5
threshold = 0.5

# Prepare training data
pretrained_model, train_loader, dev_loader, test_loader = bert_init(
    train=train, dev=dev, test=test, batch_size=batch_size, emotion_num=emotion_num, my_cache_dir="./cache/")

# Initialize model
model = BertSentimentAnalysis(
    config=pretrained_model.config, num_labels=emotion_num).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
model.config.hidden_dropout_prob = dropout_prob

# if checkpoint_path exists, load checkpoint
checkpoint_path = "./save/bert_sentiment_analysis.pth"
if os.path.exists(checkpoint_path):
    print(">>>>>>>>>>>> Loading checkpoint... ")
    model.load_state_dict(torch.load(checkpoint_path))

# Initialize tensorboard
log_dir = "./logs/"
writer = SummaryWriter(log_dir=log_dir)

# Train model
for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        input_ids, attention_mask, targets = batch_data
        input_ids, attention_mask, targets = input_ids.to(
            device), attention_mask.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        targets = targets.float()
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        writer.add_scalars('loss', {"loss": loss.item()},
                           epoch * len(train_loader) + batch_idx)

    print(">>>>>>>>>>>> Saving checkpoint... ")
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    with torch.no_grad():
        correct, total, dev_loss = 0, 0, 0
        for batch_idx, batch_data in enumerate(dev_loader):
            input_ids, attention_mask, targets = batch_data
            input_ids, attention_mask, targets = input_ids.to(
                device), attention_mask.to(device), targets.to(device)
            targets = targets.float()
            logits = model(input_ids, attention_mask)
            dev_loss += criterion(logits, targets).item()

            logits = torch.sigmoid(logits)
            predictions = (logits > threshold)
            targets = targets.bool()
            correct += torch.sum(torch.all(torch.eq(predictions, targets), dim=1))

            # print("logits : ", logits)
            # print("targets : ", targets)
            # print("predictions : ", predictions)

            total += targets.size(0)
            print("total : {}/{}, current acc : {:.2%}".format(total,
                  dev_loader.__len__() * batch_size, correct/total))

        dev_acc = correct / total
        dev_loss /= len(dev_loader)

        print(
            f"Epoch {epoch+1}, Dev Loss: {dev_loss:.2f}, Dev Acc: {dev_acc:.2%}")
