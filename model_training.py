# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

""" Choose dataset and model first! Set them in main function (lines 137, 138) """

# Custom dataset class that works for both datasets
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        """ signals: NumPy array with either shape (num_samples, segment_length) [MIT-BIH] or (num_samples, timesteps, channels) [PTB-XL].
        labels: NumPy array with shape (num_samples,) containing integer labels. """

        self.signals = signals
        self.labels = labels
        
        if len(self.signals.shape) == 2: # Add channel dimension: result -> (num_samples, 1, segment_length)
            self.signals = self.signals[:, np.newaxis, :]
        elif len(self.signals.shape) == 3:  # Transpose to (num_samples, channels, timesteps)
            self.signals = self.signals.transpose(0, 2, 1)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

# CNN Classifier
class CNNClassifier(nn.Module):
    def __init__(self, input_length, input_channels, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        L1 = (input_length - 5 + 1) // 2
        L2 = (L1 - 3 + 1) // 2
        flattened_length = L2 * 64
        self.fc1 = nn.Linear(flattened_length, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, channels, timesteps) --> LSTM expects (batch, timesteps, features)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_length, input_channels, num_classes):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()
        hidden_dim = 256
        self.fc1 = nn.Linear(input_channels * input_length, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        history.append(avg_loss)
    
    plt.plot(np.arange(1, num_epochs+1), np.array(history), label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Training Epoch on PTB-XL dataset')
    plt.legend()
    plt.show()

    return model


if __name__ == "__main__":
    """ Main training script. Initialize dataset type and model type first before use."""

    dataset_type = 'mitbih'  # Choose dataset type: 'mitbih' or 'ptbxl'
    model_type = 'cnn'       # Choose model type: 'cnn', 'lstm', 'mlp'
    num_epochs = 10          # Choose as desired, default is 10 epochs
    
    if dataset_type == 'mitbih':
        signals = np.load('mitbih_signals.npy') 
        labels = np.load('mitbih_labels.npy')
        input_length = signals.shape[1]  # e.g. 360 samples per segment
        input_channels = 1  # Filtered out only 1 channel
    elif dataset_type == 'ptbxl':
        signals = np.load('ptbxl_signals.npy')
        labels = np.load('ptbxl_labels.npy')
        input_length = signals.shape[1]  # e.g., 5000 timesteps
        input_channels = signals.shape[2]  # e.g., 12 leads
    
    num_classes = len(np.unique(labels))
    
    # Create the dataset and split into training and testing sets
    dataset = ECGDataset(signals, labels)
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'cnn':
        model = CNNClassifier(input_length=input_length, input_channels=input_channels, num_classes=num_classes)
    elif model_type == 'lstm':
        hidden_size = 64
        num_layers = 2
        model = LSTMClassifier(input_size=input_channels, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    elif model_type == 'mlp':
        model = MLPClassifier(input_length=input_length, input_channels=input_channels, num_classes=num_classes)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Change if needed
    
    # Train the model
    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    
    # Save the trained model
    model_path = f"{dataset_type}_{model_type}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}.")
