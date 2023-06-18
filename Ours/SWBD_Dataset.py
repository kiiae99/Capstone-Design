import os
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
import re
from sentiment import makesentimentdic

class SWBD_Dataset(Dataset):
    def __init__(self, path, tokenizer, length :float = 1.5) -> None:
        super().__init__()
        # self.path = os.path.join(path, "ETRI_Backchannel_Corpus_2022")
        # 
        self.tokenizer = tokenizer
        self.length = length
        self.sentiment_dict = makesentimentdic()
        self.target_sr = 16000
        self.type = path
        # balance
#        self.path = os.path.join("/local_datasets/BC/audio_equal/")
#        self.annotation=pd.read_csv('/data/minjae/BC/swbd_equal.tsv',delimiter='\t',encoding='utf-8')
        # imbalance
#        self.path = os.path.join("/local_datasets/BC/audio/")        
#        self.annotation=pd.read_csv('/data/minjae/BC/swbd.tsv',delimiter='\t',encoding='utf-8')
        if path == 'balance':
            print("Load SWBD_Dataset(balance)...")
            self.path = os.path.join("/local_datasets/BC/audio_equal/")
            self.annotation=pd.read_csv('/data/minjae/BC/swbd_equal.tsv',delimiter='\t',encoding='utf-8')
        elif path == 'imbalance':  
            print("Load SWBD_Dataset(imbalance)...")  
            self.path = os.path.join("/local_datasets/BC/audio/")        
            self.annotation=pd.read_csv('/data/minjae/BC/swbd.tsv',delimiter='\t',encoding='utf-8')
            

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ret = {}

        item = self.annotation.iloc[index]
        # print(item)
        if self.type == 'balance':
            idx = item['index'] # balance
        elif self.type == 'imbalance':
            idx = index         # imbalance
        trans = item['transcript']
        label = item['BC']
        start = item['start']
        end   = item['end']
        role  = item['role']
        role  = role == 'B'

        path = os.path.join(self.path, f"{str(idx)}.wav")
        
        audio, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        if audio.size(1)>0:
            audio = resampler(audio)
        
        audio = audio[role:role+1, -int(self.length*self.target_sr):]
        if audio.size(1) != int(self.target_sr * 1.5):
            audio = F.pad(audio, (0, int(self.target_sr * 1.5) - audio.size(1)), "constant", 0)
        

        sentiment = torch.zeros(5)
        for word in trans.split(' '):
            if word in self.sentiment_dict:
                sentiment[int(self.sentiment_dict[word])] += 1
            else:
                sentiment[0] += 1
        sentiment = sentiment / sentiment.sum()
   
        # Tokenize with padding into 64
    #    trans = self.tokenizer(trans, padding='max_length', max_length=10, truncation=True, return_tensors="pt")['input_ids'].squeeze()
        trans = self.tokenizer(trans, padding='max_length', max_length=10, truncation=True, return_tensors="pt")['input_ids'].squeeze()
        
        ret['audio'] = audio
        ret['label'] = label
        ret['text'] = trans
        ret['sentiment'] = sentiment
        return ret