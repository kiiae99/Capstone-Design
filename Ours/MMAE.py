import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(0.5) 
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Linear(d_model, d_model * 4)
        self.ffn_2 = nn.Linear(d_model * 4, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x = self.norm_1(x)
        y = self.norm_1(y)
        x2, _ = self.self_attn(x, y, y)
        x = x + x2
        
        x = self.norm_2(x)
        x2 = self.ffn_1(self.dropout(x))
        x2 = F.relu(x2)
        x2 = self.ffn_2(self.dropout(x2))
        x = x + x2
        return x

class MMAE(nn.Module):
    def __init__(self, language_model=None, audio_model=None, output_size=128, num_class=4, dropout=0.5):
        super(MMAE, self).__init__()

        self.register_module("language_model", language_model)

        assert self.language_model is not None, "bert and vocab must be provided" 

        self.register_module("audio_model", audio_model)

        self.audio_feature_size = audio_model.get_feature_size()    

        self.dropout = nn.Dropout(dropout)

        ## AVGPOOL
        self.fc_layer_1 = nn.Linear(768, 128)    
        
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class) 

        self.mae = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True) for _ in range(8)]) 

        self.audio_mask = nn.Parameter(torch.zeros(1, 1, 768))  
        self.text_mask = nn.Parameter(torch.zeros(1, 1, 768))   

        self.audio_cross1 = CrossAttentionLayer(d_model=768, nhead=12)
        self.text_cross1 = CrossAttentionLayer(d_model=768, nhead=12)   

    
        self.audio_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        self.text_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        
        self.full_downproject = nn.Linear(768, 192)
        self.audio_downproject = nn.Linear(768, 192)   
        self.text_downproject = nn.Linear(768, 192)   
        
        self.audio_predictor = nn.Linear(192*74, 24000)
        self.text_predictor = nn.Linear(192, 30522)
        
    def pretext_forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        
        original_audio = audio.clone()
        original_text = text.clone()
        
        batch_size = audio.size(0)
        
        y = {}
        
        text = self.language_model(text).last_hidden_state  # B, 64, 768 
        audio = self.audio_model(audio)                     # B, 74, 768
        
        masked_text = torch.rand_like(text.mean(-1))<0.15
        masked_audio = torch.rand_like(audio.mean(-1))<0.15
        
        text[masked_text] = self.text_mask
        audio[masked_audio] = self.audio_mask
        
        full = torch.cat((text, audio), dim = 1)
        full = self.mae(full)
        
        encoded_text = full[:,:10,:]
        encoded_audio = full[:,10:,:]    
        
        text_ = encoded_text.clone()
        audio_ = encoded_audio.clone()       
        
        encoded_text = self.text_cross1(encoded_text, audio_)
        encoded_audio = self.audio_cross1(encoded_audio, text_)
        
        encoded_audio = self.audio_downproject(encoded_audio)
        encoded_text = self.text_downproject(encoded_text)
        
        full = self.full_downproject(full)
        
        for layer in self.audio_decoder:
            encoded_audio = layer(encoded_audio, full)  
        
        for layer in self.text_decoder:
            encoded_text = layer(encoded_text, full)  
    
        audio = self.audio_predictor(encoded_audio.flatten(start_dim=1))
        text = self.text_predictor(encoded_text)

        self.pretext_loss = F.mse_loss(audio, original_audio) + F.cross_entropy(text.transpose(-1,-2), original_text)
        
        return self.pretext_loss

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]

        text = self.language_model(text).last_hidden_state
        audio = self.audio_model(audio)
        
        y = {}

        x = torch.cat((text, audio), dim=1)
        x = self.mae(x)
        
        encoded_text = x[:,:10,:]
        encoded_audio = x[:,10:,:]   
        
        text_ = encoded_text.clone()
        audio_ = encoded_audio.clone()   
        
        encoded_text = self.text_cross1(encoded_text, audio_)
        encoded_audio = self.audio_cross1(encoded_audio, text_)
        
        x = torch.cat((encoded_text, encoded_audio), dim=1)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(84,1))
        x = x.squeeze(1)
        

        x = self.fc_layer_1(self.dropout(x))
        x = self.relu(x)

        y["logit"] = self.classifier(self.dropout(x))
        
        return y