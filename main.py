import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
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
exp_name = "demo"
hypara = try_load_hypara(exp_name)
if hypara is None:
    print("No such experiment!")
    exit()

# load name
model_name = hypara['model_name']

# emotion list
emo_list_path = hypara['emo_list_path']
with open(emo_list_path, "r") as f:
    emo_list = [line.strip() for line in f.readlines()]
emotion_num = len(emo_list)

# set hyperparameters
batch_size = hypara['batch_size']
max_epochs = hypara['max_epochs']
max_epochs_after_freeze = hypara['max_epochs_after_freeze']
fixed_lr = hypara['fixed_lr']
dynamic_lr = hypara['dynamic_lr']
drpout = hypara['drpout']
fixed_threshold = hypara['threshold']
freeze_flag = hypara['freeze'] == "True"
is_fixed_threshold = hypara['is_fixed_threshold'] == "True"
emo_type = hypara['emo_type']
extended_flag = hypara['extended_flag'] == "True"
dense_num = hypara['dense_num']

print(">>>>>>>>>>>> Dense layer number is: ", dense_num)

print(">>>>>>>>>>>> The emotion list is: ", emo_list)

# load data
train, dev, test = load_data(emo_type=emo_type, extended_flag=extended_flag)

# load already runned epochs
runned_epochs = len(load_res(exp_name, emo_list))

# Prepare training data
pretrained_model, train_loader, dev_loader, test_loader = bert_init(
    train=train, dev=dev, test=test, batch_size=batch_size, emotion_num=emotion_num, my_cache_dir="./cache/")

# Initialize model
model = BertSentimentAnalysis(hidden_size=pretrained_model.config.hidden_size, dropout_prob=drpout,
                              pretrained_model=pretrained_model, num_labels=emotion_num, dense_num=dense_num).to(device)
optimizer = optim.AdamW(model.parameters(), lr=fixed_lr)
batch_nums = len(train_loader)
criterion = nn.BCEWithLogitsLoss()

if dynamic_lr == "cyclic":
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer=optimizer, base_lr=1e-5, max_lr=5e-5, step_size_up=batch_nums * 2 / 3, step_size_down=batch_nums * 4 / 3, mode="triangular2", cycle_momentum=False)
elif dynamic_lr == "constant":
    scheduler = optim.lr_scheduler.ConstantLR(optimizer=optimizer)
elif dynamic_lr == "step":
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=batch_nums, gamma=0.8)

# load fine-tuned model
if hypara['bert_model'] != "None":
    print(">>>>>>>>>>>> Loading fine-tuned model... ")
    bert_path = saving_dir + 'bert_model/' + hypara['bert_model'] + '.pth'
    model.bert.load_state_dict(torch.load(bert_path))

if freeze_flag:
    print(">>>>>>>>>>>> Freezing bert model... ")
    for param in model.bert.parameters():
        param.requires_grad = False

# if checkpoint_path exists, load checkpoint
checkpoint_path = saving_dir + exp_name + "/chkpt.pth"
if os.path.exists(checkpoint_path):
    print(">>>>>>>>>>>> Loading checkpoint... ")
    model.load_state_dict(torch.load(checkpoint_path))
    # dump a bert model
    # torch.save(model.bert.state_dict(), saving_dir + exp_name + "/bert-chkpt1-in-exp-17.pth")

# Initialize tensorboard
logs_dir = "./logs/"
writer = SummaryWriter(log_dir=logs_dir + exp_name)

# Train model


def train_model(runned_epochs, max_epochs):
    # each label has its own threshold, initialize with 0
    proper_tr = torch.tensor([0.5] * emotion_num).to(device)

    for epoch in range(runned_epochs, max_epochs):

        all_target = torch.tensor([]).to(device)
        all_logits = torch.tensor([]).to(device)

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
            prediction = torch.sigmoid(logits) > fixed_threshold

            targets = targets.bool()
            all_target = torch.cat((all_target, targets), dim=0)
            all_logits = torch.cat(
                (all_logits, torch.sigmoid(logits.detach())), dim=0)

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

        for i in range(emotion_num):
            prs, rcs, trs = precision_recall_curve(
                all_target[:, i].cpu().numpy(), all_logits[:, i].cpu().numpy())
            rcs += 1e-10
            f1_with_trs = 2 * prs * rcs / (prs + rcs)
            proper_tr[i] = torch.tensor(trs[np.argmax(f1_with_trs)])

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
                if is_fixed_threshold:
                    predictions = (logits > fixed_threshold)
                else:
                    predictions = (logits > proper_tr)
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

            f1s = f1_score(target_all_labels, pred_all_labels, average=None)
            precisions = precision_score(
                target_all_labels, pred_all_labels, average=None, zero_division=0)
            recalls = recall_score(
                target_all_labels, pred_all_labels, average=None, zero_division=0)

            res_df = load_res(exp_name, emo_list)
            dict = {}
            dict['acc'] = dev_acc.cpu().numpy()

            dict['f1_macro'] = f1_score(
                target_all_labels, pred_all_labels, average='macro')

            dict['precise_macro'] = precision_score(
                target_all_labels, pred_all_labels, average='macro', zero_division=0)

            dict['recall_macro'] = recall_score(target_all_labels, pred_all_labels,
                                                average='macro', zero_division=0)

            dict['f1_weighted'] = f1_score(
                target_all_labels, pred_all_labels, average='weighted')

            dict['precise_weighted'] = precision_score(
                target_all_labels, pred_all_labels, average='weighted', zero_division=0)

            dict['recall_weighted'] = recall_score(target_all_labels, pred_all_labels,
                                                   average='weighted', zero_division=0)

            for emo in emo_list:
                dict['f1_' + emo] = f1s[emo_list.index(emo)]

            for emo in emo_list:
                dict['precise_' + emo] = precisions[emo_list.index(emo)]

            for emo in emo_list:
                dict['recall_' + emo] = recalls[emo_list.index(emo)]

            for emo in emo_list:
                dict['tr_' +
                     emo] = proper_tr[emo_list.index(emo)].cpu().numpy()

            res_df = res_df.append(dict, ignore_index=True)

            store_res(exp_name=exp_name, df=res_df)

            print(
                f"Epoch {epoch+1}, Dev Loss: {dev_loss:.8f}, Dev Acc: {dev_acc:.4%}, F1 Score Macro: {dict['f1_macro']:.4%}, F1 Score Weighted: {dict['f1_weighted']:.4%}")


if freeze_flag:
    train_model(runned_epochs=runned_epochs,
                max_epochs=max_epochs_after_freeze)
else:
    train_model(runned_epochs=runned_epochs, max_epochs=max_epochs)
