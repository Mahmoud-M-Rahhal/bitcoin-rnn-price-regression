import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score

torch.manual_seed(42)

# Load dataset
coin_data = pd.read_csv(r"coin_Bitcoin.csv")
x = coin_data[["High", "Low", "Open"]]
y = coin_data[["Close"]]

# Standardize features and target
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)


# Split train and test data
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)


# Convert arrays to tensors
train_x = torch.tensor(train_x, dtype = torch.float32).unsqueeze(1)
train_y = torch.tensor(train_y, dtype = torch.float32)
test_x = torch.tensor(test_x, dtype = torch.float32).unsqueeze(1) 
seq_len = train_x[0].shape[0]


# Define custom dataset
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        self.x = train_x
        self.y = train_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Set training hyperparameters
hidden_size = 128
num_layers = 3
learning_rate = 0.0001
batch_size = 64
epoch_size = 8

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size = batch_size)


# Define RNN model
class RNN(nn.Module):

    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_feature_size, hidden_size, num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,1)
    
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        output = self.fc1(hidden[-1])
        output = self.relu(output)
        output = self.fc2(output)
        return output


device = torch.device("cpu")

# Initialize model, loss, and optimizer
rnn = RNN(input_feature_size = 3, hidden_size = hidden_size, num_layers = num_layers)
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)



# Train model
rnn.train()
for epoch in range(epoch_size):

    loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data
        inputs.to(device)
        targets.to(device)

        optimizer.zero_grad()

        forward = rnn.forward(inputs)
        loss = criteria(forward, targets)
        loss.backward()
        optimizer.step()

        loss += loss
        if batch_idx % 100 == 99:
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')



prediction = []
ground_truth = []
# Evaluate on test set
rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs = data[0]
        targets = data[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out


# Reverse normalization for evaluation
prediction = scaler_y.inverse_transform([[i] for i in prediction])
ground_truth = scaler_y.inverse_transform([[i] for i in ground_truth])

# Compute regression metric
r2score = r2_score(prediction,ground_truth)
print(r2score)
