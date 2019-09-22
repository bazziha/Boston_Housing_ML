# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:53:19 2019

@author: tony
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(torch.nn.Module):
    def __init__(self, n_features, size_hidden, n_output):
        super(SimpleNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, size_hidden)
        self.predict = torch.nn.Linear(size_hidden, n_output)
        
    def forward(self, x):
        x= F.relu(self.hidden(x))
        x=self.predict(x)
        return x