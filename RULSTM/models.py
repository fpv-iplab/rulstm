from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F

class OpenLSTM(nn.Module):
    """"An LSTM implementation that returns the intermediate hidden and cell states.
    The original implementation of PyTorch only returns the last cell vector.
    For RULSTM, we want all cell vectors computed at intermediate steps"""
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(OpenLSTM, self).__init__()

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        # manually iterate over each input to save the individual cell vectors
        last_cell=None
        last_hid=None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i,...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid,last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0),  torch.stack(cell, 0)

class RULSTM(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1, 
            sequence_completion=False, return_context=False):
        """
            num_class: number of classes
            feat_in: number of input features
            hidden: number of hidden units
            dropout: dropout probability
            depth: number of LSTM layers
            sequence_completion: if the network should be arranged for sequence completion pre-training
            return_context: whether to return the Rolling LSTM hidden and cell state (useful for MATT) during forward
        """
        super(RULSTM, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.rolling_lstm = OpenLSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth>1 else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))
        self.sequence_completion = sequence_completion
        self.return_context = return_context

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        inputs=inputs.permute(1,0,2)

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        x, c = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous() # batchsize x timesteps x hidden
        c = c.contiguous() # batchsize x timesteps x hidden

        # accumulate the predictions in a list
        predictions = [] # accumulate the predictions in a list
        
        # for each time-step
        for t in range(x.shape[0]):
            # get the hidden and cell states at current time-step
            hid = x[t,...]
            cel = c[t,...]

            if self.sequence_completion:
                # take current + future inputs (looks into the future)
                ins = inputs[t:,...]
            else:
                # replicate the current input for the correct number of times (time-steps remaining to the beginning of the action)
                ins = inputs[t,...].unsqueeze(0).expand(inputs.shape[0]-t+1,inputs.shape[1],inputs.shape[2]).to(inputs.device)
            
            # initialize the LSTM and iterate over the inputs
            h_t, (_,_) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            # get last hidden state
            h_n = h_t[-1,...]

            # append the last hidden state to the list
            predictions.append(h_n)
        
        # obtain the final prediction tensor by concatenating along dimension 1
        x = torch.stack(predictions,1)

        # apply the classifier to each output feature vector (independently)
        y = self.classifier(x.view(-1,x.size(2))).view(x.size(0), x.size(1), -1)
        
        if self.return_context:
            # return y and the concatenation of hidden and cell states 
            c=c.squeeze().permute(1,0,2)
            return y, torch.cat([x, c],2)
        else:
            return y

class RULSTMFusion(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
        """
        super(RULSTMFusion, self).__init__()
        self.branches = nn.ModuleList(branches)

        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        in_size = 2*len(self.branches)*hidden
        
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size,int(in_size/4)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/4), int(in_size/8)),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int(in_size/8), len(self.branches)))


    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        scores, contexts = [], []

        # for each branch
        for i in range(len(inputs)):
            # feed the inputs to the LSTM and get the scores and context vectors
            s, c = self.branches[i](inputs[i])
            scores.append(s)
            contexts.append(c)

        context = torch.cat(contexts, 2)
        context = context.view(-1, context.shape[-1])

        # Apply the MATT network to the context vectors
        # and normalize the outputs using softmax
        a = F.softmax(self.MATT(context),1)

        # array to contain the fused scores
        sc = torch.zeros_like(scores[0])

        # fuse all scores multiplying by the weights
        for i in range(len(inputs)):
            s = (scores[i].view(-1,scores[i].shape[-1])*a[:,i].unsqueeze(1)).view(sc.shape)
            sc += s

        # return the fused scores
        return sc
