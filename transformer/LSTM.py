''' Define the LSTM model '''
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Modules import Linear, SELayer, Swish

class LSTM(nn.Module):
    def __init__(self, n_trg_vocab = 43, 
                    d_words = 512, 
                    hidden_size = 512,
                    num_layers = 2):
        super().__init__()
        self.lstm = torch.nn.LSTM(d_words, hidden_size, num_layers = num_layers, batch_first = True, bidirectional = True)
        self.linear = torch.nn.Linear(hidden_size*2, n_trg_vocab)
        self.sub_rate = 4

    def forward(self, x):
        out, _ = self.lstm(x)
        # subsampling
        if out.shape[1]%self.sub_rate != 0:
            shape = out.shape
            out = out[:,:shape[1]-shape[1]%self.sub_rate,:]
        out = torch.split(out, self.sub_rate, dim=1)
        out = [torch.mean(out_enc, dim=1, keepdim=True) for out_enc in out]
        out = torch.cat(out, dim=1)

        out = self.linear(out)
        out = F.log_softmax(out, dim=2)

        return out


class LingLSTM(nn.Module):
    def __init__(self, n_trg_vocab = 43, 
                    d_words = 512, 
                    hidden_size = 512,
                    num_layers = 2):
        super().__init__()
        self.code_emb = torch.nn.Embedding(n_trg_vocab, d_words)
        self.head = LSTM(n_trg_vocab=n_trg_vocab,
                            d_words=d_words,
                            hidden_size=hidden_size,
                            num_layers=num_layers
                            )

    def forward(self, x):
        v_emb = self.code_emb(x)
        return self.head(v_emb)


# encoder-decoder scheme seq2seq network

class Encoder(nn.Module):
    def __init__(self,
                 input_size = 512,
                 hidden_size = 512,
                 n_layers = 2,
                 dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers = n_layers, dropout = dropout, batch_first = True)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        # embedded = self.dropout(F.relu(self.linear(x)))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(x)
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                 output_size = 43,
                 embedding_size = 512,
                 hidden_size = 512,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(embedding_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout, batch_first = True)
        # self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        # x = x.unsqueeze(1)

        # embedded = [batch size, 1, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x)))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hidden size]
        #hidden = [n layers, batch size, hidden size]
        #cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        # prediction = self.linear(output.squeeze(0))

        # return prediction, hidden, cell
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, n_trg_vocab = 43, 
                    d_words = 512, 
                    hidden_size = 512,
                    num_layers = 2):
        super().__init__()
        self.code_emb = torch.nn.Embedding(n_trg_vocab, d_words)
        self.projection =  nn.Sequential(
            nn.LayerNorm(d_words),
            Swish(),
            Linear(d_words, d_words)#,
            # nn.LayerNorm(d_words),
            # Swish(),
            # Linear(d_words, d_words), # add 1 layer to projection
            )
        self.head = Seq2SeqHead()

    def forward(self, x, y, target_len, teacher_forcing_ratio = 0.5):
        x = self.code_emb(x)
        y = self.code_emb(y)
        x = self.projection(x)
        y = self.projection(y)
        return self.head(x, y, target_len, teacher_forcing_ratio)

        

class Seq2SeqHead(nn.Module):
    # def __init__(self, encoder, decoder, device):
    def __init__(self, n_trg_vocab = 43, 
                    d_words = 512, 
                    hidden_size = 512,
                    num_layers = 2):
        super().__init__()
        # self.code_emb = torch.nn.Embedding(n_trg_vocab, d_words)
        self.layernorm = nn.LayerNorm(d_words)
        self.encoder = Encoder(input_size=d_words, hidden_size=hidden_size, n_layers=num_layers, dropout=0.5)
        self.decoder = Decoder(output_size=n_trg_vocab, embedding_size=hidden_size, hidden_size=hidden_size, n_layers=num_layers, dropout=0.5)
        self.linear = nn.Linear(hidden_size, n_trg_vocab)
        
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0')
        # self.device = torch.device('cpu')

        # assert encoder.hidden_size == decoder.hidden_size, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, target_len, teacher_forcing_ratio = 0.5):
        """
        x = [batch size, seq_len]
        y = [batch size, seq_len]
        for our argoverse motion forecasting dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        # batch_size = x.shape[1]
        # target_len = y.shape[1]
        # y = self.code_emb(y)
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape[0], target_len, self.hidden_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # x = self.code_emb(x)
        x = self.layernorm(x)
        y = self.layernorm(y)

        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = x[:, -1::1, :]
        
        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[:, i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[:, i:i+1, :] if teacher_forcing else output

        feature = outputs
        outputs = self.linear(outputs)
        outputs = F.log_softmax(outputs, dim=2)

        return feature, outputs

if __name__ == "__main__":
    # lstm = LingLSTM()
    lstm = Seq2Seq()
    test_ts = torch.tensor([ 3, 32, 41, 22, 34, 41, 14, 22, 38, 41, 14, 22, 38, 41,  5, 27, 41,  7, 25, 41,  3, 27, 41, 11, 33])
    test_ts = test_ts.unsqueeze(0)
    feature, out = lstm(test_ts, test_ts, 6)
    # out = lstm(test_ts)
    print(feature.shape)
    print(out.shape)