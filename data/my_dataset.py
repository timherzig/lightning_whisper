import audiofile
import audresample
import pandas as pd

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio, sr = audiofile.read(row['path'])

        audio = audresample.resample(audio, sr, 16000)

        return audio, 16000, row['label']

    def __len__(self):
        return len(self.df)