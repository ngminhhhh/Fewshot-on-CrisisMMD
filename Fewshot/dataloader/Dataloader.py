import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class CrisisDataset(Dataset):
    def __init__(self, tsv_file, image_dir, preprocess):
        super().__init__()

        self.df = pd.read_csv(tsv_file, sep="\t")
        self.image_dir = image_dir
        self.preprocess = preprocess

        self.df = self.df[["tweet_text", "image", "label"]]
        self.df = self.df.dropna().reset_index(drop=True)
        
        labels = sorted(self.df["label"].unique())
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tweet_text = str(row["tweet_text"])
        image_name = str(row["image"])
        label_name = row["label"]
        label_id = self.label2id[label_name]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        return {
            "image": image,
            "text": tweet_text,
            "label": label_id
        }