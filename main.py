import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from settings import load_res, store_res, store_train, try_load_hypara
from model import BertSentimentAnalysis
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loader import load_data
from bertpre import bert_init

# Set device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load settings
print(">>>>>>>>>>>> Loading settings... ")
saving_dir = "./saving/"
exp_name = "exp-0"
hypara = try_load_hypara(exp_name)
if hypara is None:
    print("No such experiment!")
    exit()

# load name
exp_name = hypara['exp_name']
model_name = hypara['model_name']

# emotion list
emo_list_path = hypara['emo_list_path']
with open(emo_list_path, "r") as f:
    emo_list = [line.strip() for line in f.readlines()]
emotion_num = len(emo_list)

# set hyperparameters
batch_size = hypara['batch_size']
max_epochs = hypara['max_epochs']
fixed_lr = hypara['fixed_lr']
dynamic_lr = hypara['dynamic_lr']
drpout = hypara['drpout']
threshold = hypara['threshold']

# load data
train, dev, test = load_data()

# load already runned epochs
runned_epochs = len(load_res(exp_name, emo_list))

# Prepare training data
pretrained_model, train_loader, dev_loader, test_loader = bert_init(
    train=train, dev=dev, test=test, batch_size=batch_size, emotion_num=emotion_num, my_cache_dir="./cache/")

# initialize dropout prob
pretrained_model.config.hidden_dropout_prob = drpout
# Initialize model
model = BertSentimentAnalysis(
    config=pretrained_model.config, num_labels=emotion_num).to(device)
optimizer = optim.AdamW(model.parameters(), lr=fixed_lr)
batch_nums = len(train_loader)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ConstantLR(optimizer=optimizer)
# scheduler = optim.lr_scheduler.CyclicLR(
#     optimizer=optimizer, base_lr=1e-5, max_lr=5e-5, step_size_up=batch_nums * 2 / 3, step_size_down=batch_nums * 4 / 3, mode="triangular2", cycle_momentum=False)

# if checkpoint_path exists, load checkpoint
checkpoint_path = saving_dir + exp_name + "/chkpt.pth"
if os.path.exists(checkpoint_path):
    print(">>>>>>>>>>>> Loading checkpoint... ")
    model.load_state_dict(torch.load(checkpoint_path))

# Initialize tensorboard
logs_dir = "./logs/"
writer = SummaryWriter(log_dir=logs_dir + exp_name)

# Train model
for epoch in range(runned_epochs, max_epochs):
    model.train()
    train_df = pd.DataFrame(columns=['loss', 'lr', 'acc'])
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
        scheduler.step()

        # calculate accuracy on training set
        prediction = torch.sigmoid(logits) > threshold
        targets = targets.bool()
        acc_on_train = torch.sum(
            torch.all(torch.eq(prediction, targets), dim=1)) / targets.size(0)

        train_df.loc[batch_idx] = [
            loss.item(), optimizer.param_groups[0]['lr'], acc_on_train.cpu().numpy()]

        print(
            f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.8f}, Acc: {acc_on_train:.8f}")

        writer.add_scalars('loss', {"loss": loss.item()},
                           epoch * len(train_loader) + batch_idx)
        writer.add_scalars(
            'lr', {"lr": optimizer.param_groups[0]['lr']}, epoch * len(train_loader) + batch_idx)
        writer.add_scalars(
            'acc_train', {"acc_train": acc_on_train}, epoch * len(train_loader) + batch_idx)

    store_train(exp_name, epoch, train_df)

    print(">>>>>>>>>>>> Saving checkpoint... ")
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    with torch.no_grad():
        correct, total, dev_loss = 0, 0, 0
        target_all_labels, pred_all_labels = [], []

        for batch_idx, batch_data in enumerate(dev_loader):
            input_ids, attention_mask, targets = batch_data
            input_ids, attention_mask, targets = input_ids.to(
                device), attention_mask.to(device), targets.to(device)
            targets = targets.float()
            logits = model(input_ids, attention_mask)
            current_loss = criterion(logits, targets).item()
            dev_loss += current_loss

            logits = torch.sigmoid(logits)
            predictions = (logits > threshold)
            targets = targets.bool()

            pred_labels = predictions.cpu().numpy()
            target_labels = targets.cpu().numpy()

            target_all_labels.extend(target_labels)
            pred_all_labels.extend(pred_labels)

            # all labels are correct
            correct += torch.sum(torch.all(torch.eq(predictions, targets), dim=1))

            total += targets.size(0)
            print("total : {}/{}, current acc : {:.2%}, current loss : {:.4}".format(total,
                  dev_loader.__len__() * batch_size, correct/total, current_loss))

        dev_acc = correct / total
        dev_loss /= len(dev_loader)
        f1 = f1_score(target_all_labels, pred_all_labels, average='macro')
        pre = precision_score(
            target_all_labels, pred_all_labels, average='macro', zero_division=0)
        rec = recall_score(target_all_labels, pred_all_labels,
                           average='macro', zero_division=0)

        f1s = f1_score(target_all_labels, pred_all_labels, average=None)
        precisions = precision_score(
            target_all_labels, pred_all_labels, average=None, zero_division=0)
        recalls = recall_score(
            target_all_labels, pred_all_labels, average=None, zero_division=0)

        res_df = load_res(exp_name, emo_list)
        dict = {}
        dict['acc'] = dev_acc.cpu().numpy()
        dict['f1'] = f1
        dict['precise'] = pre
        dict['recall'] = rec
        for emo in emo_list:
            dict['f1_' + emo] = f1s[emo_list.index(emo)]
            dict['precise_' + emo] = precisions[emo_list.index(emo)]
            dict['recall_' + emo] = recalls[emo_list.index(emo)]

        res_df = res_df.append(dict, ignore_index=True)

        store_res(exp_name=exp_name, df=res_df)

        print(
            f"Epoch {epoch+1}, Dev Loss: {dev_loss:.8f}, Dev Acc: {dev_acc:.4%}, F1 Score: {f1:.4%}")
