from torch import nn

class RNN_Final(nn.Module):
    def __init__(self, interval, num_pc, ch_by_bands_num=5):
        super(RNN_Final, self).__init__()
        self.num_pc = num_pc
        self.linear = nn.Linear(ch_by_bands_num, 16)
        self.bn = nn.BatchNorm1d(16 * interval)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        hidden_size = 8
        self.rnn1 = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bn1 = nn.BatchNorm1d(num_features=interval * hidden_size * 2 * 1)
        self.dropout1 = nn.Dropout(0.3)
        self.out1 = nn.Sequential(nn.Linear(interval * hidden_size * 2, num_pc),)


    def forward(self, x):
        """
        x (batch, time_step, input_size)
        """
        batch_size = x.shape[0]
        time_step = x.shape[1]

        x_out = self.linear(x)
        x_out = self.relu(self.bn(x_out.view(batch_size, -1)))
        x_out = self.dropout(x_out)
        x_out = x_out.view(batch_size, time_step, -1)
        r_out, (h_state, c_state) = self.rnn1(x_out, None)
        r_out_intermediate = r_out.contiguous().view(batch_size, -1)
        r_out_intermediate = self.bn1(r_out_intermediate)

        out = self.out1(r_out_intermediate)
        return out


