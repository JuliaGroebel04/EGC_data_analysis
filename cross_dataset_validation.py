# cross_dataset_validation.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from model_training import ECGDataset, CNNClassifier, LSTMClassifier, MLPClassifier


""" Choose test-dataset and model first! Set them here: """
# For example, pretrained_model from MIT_BIH and testing on PTB-XL:
test_dataset_type = 'ptbxl'        # 'mitbih' or 'ptbxl'
pretrained_dataset_type = 'mitbih'     # 'ptbxl' or 'mitbih'
model_type = 'cnn'                  # Options: 'cnn', 'lstm', or 'mlp'


# Helper function: segment signal to fixed length
def segment_signal(signal, seg_len=360):
    """ Given a 1D numpy array 'signal' of arbitrary length, extract a centered segment of length 'seg_len'.
    If the signal is shorter than seg_len, pad with zeros."""
    L = len(signal)
    if L < seg_len:
        pad_left = (seg_len - L) // 2
        pad_right = seg_len - L - pad_left
        return np.pad(signal, (pad_left, pad_right), mode='constant')
    else:
        start = (L - seg_len) // 2
        return signal[start:start+seg_len]

# Load test dataset and adjust channels
if test_dataset_type == 'mitbih':
    signals = np.load('mitbih_signals.npy')  # shape: (N, segment_length) where segment_length=360
    labels = np.load('mitbih_labels.npy')
    # Apply binary mapping: 0 = Normal, 1 = Abnormal (for MIT-BIH: 0 for 'N' and '/' are normal)
    binary_labels = np.vectorize(lambda x: 0 if x == 0 else 1)(labels)
    input_length = signals.shape[1]  # 360
    input_channels = 1
    # If pretrained model expects 12 channels (from PTB-XL), replicate the 1 channel 12 times:
    if pretrained_dataset_type == 'ptbxl':
        signals = np.expand_dims(signals, axis=2)  # (N, 360, 1)
        signals = np.repeat(signals, 12, axis=2)  # (N, 360, 12)
        input_channels = 12

elif test_dataset_type == 'ptbxl':
    signals = np.load('ptbxl_signals.npy')  # shape: (N, timesteps, channels), e.g., (N, 5000, 12)
    labels = np.load('ptbxl_labels.npy')
    binary_labels = np.vectorize(lambda x: 0 if x == 0 else 1)(labels)
    # Select one channel (e.g., the first channel)
    signals = signals[:, :, 0]  # Now shape: (N, timesteps)
    # Segment each recording to a fixed length (360 samples)
    signals = np.array([segment_signal(sig, 360) for sig in signals])
    input_length = 360  # Segments
    input_channels = 1  # Channels

dataset = ECGDataset(signals, binary_labels)
indices = list(range(len(dataset)))
_, test_indices = train_test_split(indices, test_size=0.3)
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if pretrained_dataset_type == 'ptbxl':
    # Pretrained model from PTB-XL was trained with 12 channels and 6 output classes
    pretrained_input_length = 360   # Assume a fixed-length segment (e.g., 360 samples) was used during training
    pretrained_input_channels = 12
    num_classes_pretrained = 6
elif pretrained_dataset_type == 'mitbih':
    pretrained_input_length = 360
    pretrained_input_channels = 1
    num_classes_pretrained = 5


if model_type == 'cnn':
    model = CNNClassifier(input_length=pretrained_input_length, 
                          input_channels=pretrained_input_channels, 
                          num_classes=num_classes_pretrained)
elif model_type == 'lstm':
    hidden_size = 64
    num_layers = 2
    model = LSTMClassifier(input_size=pretrained_input_channels, hidden_size=hidden_size, 
                           num_layers=num_layers, num_classes=num_classes_pretrained)
elif model_type == 'mlp':
    model = MLPClassifier(input_length=pretrained_input_length, 
                          input_channels=pretrained_input_channels, 
                          num_classes=num_classes_pretrained)
else:
    raise ValueError("Invalid model type. Choose 'cnn', 'lstm', or 'mlp'.")

model.to(device)
model_path = f"{pretrained_dataset_type}_{model_type}_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# Replace the final layer to adapt the model for binary classification.
# Note: This new layer is randomly initialized. For a proper evaluation, you might fine-tune this layer.
if model_type == 'cnn':
    model.fc2 = torch.nn.Linear(128, 2)
elif model_type == 'lstm':
    model.fc = torch.nn.Linear(hidden_size * 2, 2)
elif model_type == 'mlp':
    model.fc2 = torch.nn.Linear(256, 2)
model.to(device)
model.eval()


def evaluate_model(model, test_loader, device):
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
        auc = None
        print(f"AUC calculation error: {e}")
    
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, auc, cm


acc, f1, auc, cm = evaluate_model(model, test_loader, device)
print("Cross-Dataset Validation Results:")
print("Test Dataset:", test_dataset_type)
print("Pretrained Model from:", pretrained_dataset_type)
print("Accuracy:", acc)
print("F1 Score:", f1)
print("AUC-ROC:", auc)
print("Confusion Matrix:\n", cm)


