# model_evaluation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
# Import custom classes from the training module (check if script-name is right)
from model_training import ECGDataset, CNNClassifier, LSTMClassifier, MLPClassifier

""" Choose dataset and model first! Set them here: """

dataset_type = 'mitbih'  # 'mitbih' or 'ptbxl'
model_type = 'cnn'       # 'cnn', 'lstm', or 'mlp'

if dataset_type == 'mitbih':
    signals = np.load('mitbih_signals.npy')
    labels = np.load('mitbih_labels.npy')
    input_length = signals.shape[1]  # e.g., 360
    input_channels = 1
elif dataset_type == 'ptbxl':
    signals = np.load('ptbxl_signals.npy')
    labels = np.load('ptbxl_labels.npy')
    input_length = signals.shape[1]  # e.g., timesteps (e.g., 5000)
    input_channels = signals.shape[2]  # e.g., 12 leads

num_classes = len(np.unique(labels))

# Create dataset and split into train/test (use only test here)
dataset = ECGDataset(signals, labels)
indices = list(range(len(dataset)))
_, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
test_dataset = Subset(dataset, test_indices)
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
else:
    raise ValueError("Invalid model type. Choose 'cnn', 'lstm', or 'mlp'.")

model.to(device)

# Load the saved model; ensure the filename matches the model type
model_path = f"{dataset_type}_{model_type}_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def evaluate_model(model, test_loader, device):
    """ Evaluate the model on the test set and compute metrics. """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for signals, labels in test_loader:
        signals = signals.to(device)
        outputs = model(signals)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.detach().cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    except Exception as e:
        auc = None  # AUC calculation may fail for multi-class problems.
    
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, auc, cm

# Evaluate the model
acc, f1, auc, cm = evaluate_model(model, test_loader, device)
print("Accuracy:", acc)
print("F1 Score:", f1)
print("AUC-ROC:", auc)
print("Confusion Matrix:\n", cm)


def compute_saliency(model, input_tensor, target_class, device):
    """ Compute a basic saliency map by taking the gradient of the output for the target class with respect to the input. """
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True
    output = model(input_tensor.unsqueeze(0))  # Add batch dimension
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()
    # For 1D signals, take the max gradient across channels
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=0)
    
    return saliency.cpu().numpy()

# Compute saliency for a sample from the test set
sample_signal, sample_label = test_dataset[0]
sample_signal = sample_signal.to(device)
predicted_class = model(sample_signal.unsqueeze(0)).argmax(dim=1).item()
saliency_map = compute_saliency(model, sample_signal, predicted_class, device)

plt.figure(figsize=(10, 3))
plt.plot(saliency_map)
plt.title("Saliency Map for a Sample ECG Signal (MIT-BIH, CNN)")
plt.xlabel("Time Steps")
plt.ylabel("Saliency")
plt.show()

