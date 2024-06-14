import torch
from torch.nn import Module
import torch.nn as nn


class LSTMagija(Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
        """
        
        X = [
            [ [x1, y1, z1], [x2, y2,z2], ... , [xn, yn, zn] ],
            [ ... ],
            
        ]
        
        x.shape = [batch][time sequence][3]
        
        
        x1 --> f1 --> y1
        
        y1 -- >f1 --> y2
               f2 --> k2
               
        y2 --> f1 --> y3
               f2 --> k3
               
               
        x -> F -> [k1, ... kn]
        
        """
        self.lstm1 = nn.LSTM(3, 128, batch_first=True)        
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.LazyLinear(2)
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        x, h = self.lstm1(x)
        x = self.feature_extractor(x[:, -1])
        x = self.softmax(x)
        return x
