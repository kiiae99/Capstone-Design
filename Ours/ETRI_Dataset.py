import os
import gc
import shutil
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
from knusl import KnuSL
import math
import numpy as np

class ETRI_Corpus_Dataset(Dataset):
    def __init__(self, path, tokenizer, transform : Callable=None, length :float = 1.5) -> None:
        super().__init__()
        # self.path = os.path.join(path, "ETRI_Backchannel_Corpus_2022")
        print("Load ETRI_Corpus_Dataset...")
        self.tokenizer = tokenizer
        self.path = os.path.join("/local_datasets/BC/etri_last/")
        self.length = length
        self.annotation = pd.read_csv('/data/minjae/BC/etri_last.tsv',delimiter='\t',encoding='utf-8')
        self.sr = 16000
        self.type = path

        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ret = {}

        item = self.annotation.iloc[index]
        # print(item)
        trans = item['transcript']
        if self.type == 'imbalance':
            if item['BC'] == 0:
                label = 0
            else: 
                label = 1
        elif self.type == 'balance':            
            label = item['BC']
        start = item['start']
        end   = item['end']
        role = item['role']
        role  = role == 1
        
        path = os.path.join(self.path, f"{str(index)}.wav")
        
    
        audio, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        
        if audio.size(1)>0:
            audio = resampler(audio)
        
        audio = audio[role:role+1, -int(self.length*self.sr):]
        
        if audio.size(1) != int(self.sr * 1.5):
            audio = F.pad(audio, (0, int(self.sr * 1.5) - audio.size(1)), "constant", 0)


        sentiment = torch.zeros(5)
        for word in trans.split(' '):
            r_word, s_word = KnuSL.data_list(word)
            if s_word != 'None':
                sentiment[int(s_word)] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
        
        trans = self.tokenizer(trans, padding='max_length', max_length=10, truncation=True, return_tensors="pt")['input_ids'].squeeze()

            
        ret['audio'] = audio
        ret['label'] = label
        ret['text'] = trans
        ret['sentiment'] = sentiment
        return ret