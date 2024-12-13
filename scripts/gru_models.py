import torch
import torch.nn as nn

class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MinGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # GRU Forward Pass
        _, hidden = self.gru(x)
        out = self.fc(hidden[-1])  # Take the last hidden state
        return out

class TraditionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TraditionalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # GRU Forward Pass
        _, hidden = self.gru(x)
        out = self.fc(hidden[-1])  # Take the last hidden state
        return out

if __name__ == "__main__":
    print("This script provides GRU model definitions and should not be run independently.")
