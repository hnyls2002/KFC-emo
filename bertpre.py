from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification


class EmotionsDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get text and label
        text = self.df.loc[idx, "text"]
        label = self.df.loc[idx, "label"]

        # Tokenize input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Return input IDs, attention mask, and label
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze(), label


def bert_init(train, dev, test, batch_size, emotion_num, my_cache_dir):
    # Load pre-trained BERT tokenizer and encoder
    print(">>>>>>>>>>>> Loading BERT tokenizer and encoder... ")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', cache_dir=my_cache_dir)
    print(">>>>>>>>>>>> Loading BERT model... ")
    pretrained_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=emotion_num, cache_dir=my_cache_dir)

    print(">>>>>>>>>>>> BERT Preparing for training... ")
    train_dataset = EmotionsDataset(tokenizer, train)
    print(">>>>>>>>>>>> BERT Preparing for dev...")
    dev_dataset = EmotionsDataset(tokenizer, dev)
    print(">>>>>>>>>>>> BERT Preparing for test...")
    test_dataset = EmotionsDataset(tokenizer, test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return pretrained_model, train_loader, dev_loader, test_loader
