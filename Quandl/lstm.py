import torch
import torch.nn as nn

from statistics import mean

import numpy as np

from tqdm import tqdm

import time

class Net(nn.Module):
    
    def __init__(self, input_size = 8, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0,
                 output_size = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.dropout = dropout
        
        #Use LSTM layer since we're dealing with sequential data
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first,
                          dropout = self.dropout)
        #Use a linear layer to determine the size of the output sequence. 
        self.fc = nn.Linear(self.hidden_size, self.output_size) 
        
        
    def forward(self, X):   
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        self.h_t = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        self.c_t = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        self.hidden_state = (self.h_t, self.c_t)

        X, self.hidden_state = self.rnn(X, self.hidden_state)
        X = X.reshape(seq_len, batch_size, -1)
        X = self.fc(X[-1])
        
        return X
       
    
def fwd_pass(X,y,model, train = False):
        
    output = model(X)
    loss = loss_function(output, y)
    
    if train:
        model.zero_grad()
        loss.backward()
        optimiser.step()
        
    return loss
    
 