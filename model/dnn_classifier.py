import torch
from torch import nn

class DNN_Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        x = torch.softmax(x, dim = 1)

        return x
    