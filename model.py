import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        # Weight matrix W
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        # Weight matrix U
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        # Bias vector
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    # Initialize weights uniformly within [-stdv,stdv]
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        # Extract batch size and sequence size
        bs, seq_sz, _ = x.size()
        # List to store output of each time step
        hidden_seq = []
        # Check if initial states are given, else states as zero
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        # Iterate through time steps in the sequence
        for t in range(seq_sz):
            # Extract input for each time step
            x_t = x[:, t, :]
            # Batch the computations into a single matrix multiplication
            # Compute gate values by applying linear transformations + bias
            gates = x_t @ self.W + h_t @ self.U + self.bias
            # Split gate values and apply activations
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            # Cell state update
            c_t = f_t * c_t + i_t * g_t
            # Hidden state update based on output gate and new cell state
            h_t = o_t * torch.tanh(c_t)
            # Append current hidden state to list
            hidden_seq.append(h_t.unsqueeze(0))
        # Concatenate hidden states along time dimension  
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        # 1D convolutional layer
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        # Xavier weight initialization good against vanishing and exploding gradients
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    # Take input and apply convolution
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



# VCModel can be initialized with boolean options more_dropout, dimincrease, postnet, use_custom_lstm
class VCModel(nn.Module):
    def __init__(self, more_dropout, dimincrease, postnet, use_custom_lstm=False):
        super().__init__()
        self.postnet = postnet
        self.encoder = Encoder(more_dropout)
        self.decoder = Decoder(more_dropout, dimincrease, use_custom_lstm=use_custom_lstm)
        if postnet:
          self.postnet = Postnet()

    # Pass content embedding x through encoder, concatenate with spk_emb and pass through decoder (and optionally postnet)
    def forward(self, x, spk_embs, mels):
        x = self.encoder(x)
        exp_spk_embs = spk_embs.unsqueeze(1).expand(-1, x.size(1), -1)
        concat_x = torch.cat([x, exp_spk_embs], dim=-1)
        output = self.decoder(concat_x, mels)
        if self.postnet:
          final_output = self.postnet(output) + output
        else:
          final_output = output
        return final_output

    # Separate generate function due to differences between training and generation
    @torch.inference_mode()
    def generate(self, x, spk_embs):
        x = self.encoder(x)
        exp_spk_embs = spk_embs.unsqueeze(1).expand(-1, x.size(1), -1)
        concat_x = torch.cat([x, exp_spk_embs], dim=-1)
        mels = self.decoder.generate(concat_x)
        # If postnet add postnet output with Decoder output
        if self.postnet:
          final_mels = self.postnet(mels) + mels
        else:
          final_mels = mels
        return final_mels


class Encoder(nn.Module):
    def __init__(self, more_dropout):
        super().__init__()
        self.prenet = PreNet(256, 256, 256)

        if more_dropout:
          self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1),
            nn.Dropout(0.3),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.InstanceNorm1d(512),
          )
        else:
          self.convs = nn.Sequential(
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.ConvTranspose1d(512, 512, 4, 2, 1),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(512),
        )

    # Pass x content embedding through 2 linear layers, apply convolutions on transposed and then transpose again.
    def forward(self, x):
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, more_dropout, dimincrease, use_custom_lstm=False):
        super().__init__()
        # Dropout option
        self.more_dropout = more_dropout
        # Hidden Dimension
        self.hidden_dim = 1024 if dimincrease else 768
        # Custom LSTM option
        self.use_custom_lstm = use_custom_lstm
        self.prenet = PreNet(128, 256, 256)
        # You can decide if you want use the custom LSTM or PyTorch LSTM
        if use_custom_lstm:
          self.lstm1 = CustomLSTM(1024 + 256, self.hidden_dim)
          self.lstm2 = CustomLSTM(self.hidden_dim, self.hidden_dim)
          self.lstm3 = CustomLSTM(self.hidden_dim, self.hidden_dim)
        else:
          self.lstm1 = nn.LSTM(1024 + 256, self.hidden_dim)
          self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim)
          self.lstm3 = nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, 128, bias=False)
        if self.more_dropout:
          self.dropout = nn.Dropout(0.3)

    def forward(self, x, mels):
        # Pass mels through prenet
        mels = self.prenet(mels)
        # Concatenate Encoder ouput with mels and pass through LSTMs
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
        if self.more_dropout:
          x = self.dropout(x)
        res = x
        x, _ = self.lstm2(x)
        if self.more_dropout:
          x = self.dropout(x)
        # Add output to previous output for a residual connection
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        if self.more_dropout:
          x = self.dropout(x)
        x = res + x
        # Linear projection to outputshape 128 (mel spectogram bins)
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        # List for storing generated outputs
        m = torch.zeros(xs.size(0), 128, device=xs.device)
        # Initialize hidden state and cell state with zeros
        if self.use_custom_lstm:
          h1 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
          c1 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
          h2 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
          c2 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
          h3 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
          c3 = torch.zeros(xs.size(0), self.hidden_dim, device=xs.device)
        else:
          h1 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)
          c1 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)
          h2 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)
          c2 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)
          h3 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)
          c3 = torch.zeros(1, xs.size(0), self.hidden_dim, device=xs.device)


        mel = []
        # Iterate over each time step, in each new time step utilizing the generations for each previous timestep
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            # Apply first and second LSTMs and add outputs for a residual connection
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            # Apply last LSTM and add outputs for a residual connection
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            # Apply linear projection and append to mel list
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)

# 2 Linear Layers with Dropout 0.5
class PreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.net(x)



class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.n_mel_channels = 128
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5
        # First Convolution with 1d Batchnorm layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(in_channels=self.n_mel_channels,  # Adjusted input channels
                         out_channels=self.postnet_embedding_dim,  # Output channels remain the same
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),  # Dynamic padding
                         dilation=1, bias=True, w_init_gain='tanh'),
                nn.BatchNorm1d(self.postnet_embedding_dim)
            )
        )
        
        # Middle Convolution with 1d Batchnorm layers
        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size, stride=1,
                             padding=int((self.postnet_kernel_size - 1) / 2),  # Dynamic padding
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.postnet_embedding_dim)
                )
            )

        # Last Convolution with 1d Batchnorm layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.postnet_embedding_dim, self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),  # Dynamic padding
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.n_mel_channels)
            )
        )


    def forward(self, x):
        x = x.transpose(1, 2)
        # Iterate over each conv layer except for the last one, apply tanh and dropout
        for i, conv in enumerate(self.convolutions[:-1]):
            x = conv(x)
            x = torch.tanh(x)
            x = F.dropout(x, 0.5, self.training)

        # Last layer with dropout
        x = self.convolutions[-1](x)
        x = F.dropout(x, 0.5, self.training)
        x = x.transpose(1, 2)

        return x