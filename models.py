import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.dp = nn.Dropout(0.1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(x.size(0), -1) #aplanar imagen
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dp(x)
    x = self.fc2(x)
    return x

class CNN(nn.Module):
  def __init__(self, output_size):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
    self.fc1 = nn.Linear(64*7*7, 128)
    self.fc2 = nn.Linear(128, output_size)
    self.maxPool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.dp = nn.Dropout(0.4)

  def forward(self, x):
    #extractor
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxPool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxPool(x)

    #clasificador
    x = x.view(x.size(0), -1) #aplanar imagen
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dp(x)
    x = self.fc2(x)
    return x
