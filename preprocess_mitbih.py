# preprocess_mitbih.py

import wfdb
import numpy as np
import os

# Ensure that you have downloaded the MIT-BIH database (https://physionet.org/content/mitdb/1.0.0/)
# Choose the right path
data_path = './mit-bih-arrhythmia-database/' 

# Define a label mapping for MIT-BIH annotations (customize as needed, a full list of labels can be seen in the documentation on physionet)
label_mapping = {
    # Normal beats
    'N': 0,  # Normal beat
    '/': 0,  # Paced beat

    # bundle brach block beats
    'L': 1,  # Left bundle branch block beat
    'R': 1,  # Right bundle branch block beat

    # Supraventricular ectopic beats
    'A': 2,  # Atrial premature beat
    'a': 2,  # Aberrated atrial premature beat
    'J': 2,  # Nodal (junctional) premature beat
    'S': 2,  # Supraventricular premature beat

    # Ventricular ectopic beats
    'V': 3,  # Ventricular ectopic beat
    'E': 3,  # Ventricular escape beat

    # Fusion beats
    'F': 4   # Fusion beat
}

record_names = [os.path.splitext(fname)[0] for fname in os.listdir(data_path) if fname.endswith('.dat')]

signals_list = []
labels_list = []
segment_length = 360  # e.g. 360 samples per segment

for rec_name in record_names:
    try:
        rec = wfdb.rdrecord(os.path.join(data_path, rec_name))
        ann = wfdb.rdann(os.path.join(data_path, rec_name), 'atr')
    except Exception as e:
        print(f"Error reading record {rec_name}: {e}")
        continue

    for i, sample in enumerate(ann.sample):
        start = sample - segment_length // 2
        end = start + segment_length
        # Skip segments that exceed signal boundaries
        if start < 0 or end > len(rec.p_signal):
            continue
        # Only append if the label is in the mapping
        label = ann.symbol[i]
        if label in label_mapping:
            segment = rec.p_signal[start:end, 0]  # Extract segment from the first channel
            signals_list.append(segment)
            labels_list.append(label_mapping[label])
        else:
            continue

signals_array = np.array(signals_list) 
labels_array = np.array(labels_list)    

# Save preprocessed data
np.save('mitbih_signals.npy', signals_array)
np.save('mitbih_labels.npy', labels_array)

print("MIT-BIH preprocessing complete. Saved 'mitbih_signals.npy' and 'mitbih_labels.npy'.")
