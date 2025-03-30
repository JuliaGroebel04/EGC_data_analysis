# preprocess_ptbxl.py

import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import LabelEncoder

# Ensure that you have downloaded the PTB-XL database (https://physionet.org/content/ptb-xl/1.0.1/)
# Choose the right path
base_path = './ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate = 100  # Use 100 or 500 based on desired resolution

def load_raw_data(df, sampling_rate, path):
    """ Load raw ECG signal data from PTB-XL.
    If sampling_rate==100, use the 'filename_lr' column; otherwise, use the 'filename_hr' column. """
    
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    # Extract only the signal part from each tuple (signal, meta)
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic, agg_df):
    """ Aggregate diagnostic information using scp_statements.
    For each key in the scp_codes dictionary, if the key is present in agg_df, append the corresponding diagnostic_class.
    Returns a list of unique diagnostic classes. """

    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

meta_path = base_path + 'ptbxl_database.csv'
Y = pd.read_csv(meta_path, index_col='ecg_id')
# Convert the 'scp_codes' column from string to dictionary
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv to aggregate diagnostic labels
agg_df = pd.read_csv(base_path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]  # Filter for diagnostic statements only

# Create the diagnostic_superclass column by aggregating scp_codes
Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
# For simplicity, if a record has multiple diagnostic superclasses, choose the first one
Y['diagnostic_superclass'] = Y['diagnostic_superclass'].apply(lambda lst: lst[0] if len(lst) > 0 else 'Undefined')


X = load_raw_data(Y, sampling_rate, base_path)
labels = Y['diagnostic_superclass'].values
le = LabelEncoder() # Optionally, filter out 'Undefined' labels or handle them accordingly
encoded_labels = le.fit_transform(labels)

# Save preprocessed data
np.save('ptbxl_signals500.npy', X)
np.save('ptbxl_labels100.npy', encoded_labels)
print("PTB-XL preprocessing complete. Saved 'ptbxl_signals.npy' and 'ptbxl_labels.npy'.")

