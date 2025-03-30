# Deep Learning in ECG Analysis: Approaches to Arrhythmia Detection
This repository contains code for preprocessing, training, evaluation, and cross-dataset validation of deep learning models for arrhythmia detection using the  datasets MIT-BIH (find here: https://physionet.org/content/mitdb/1.0.0/) and PTB-XL (find here: https://physionet.org/content/ptb-xl/1.0.1/).

## Overview
  This project implements a unified framework for arrhythmia detection that:<br>
  Preprocesses two widely-used ECG datasets (MIT-BIH and PTB-XL) by extracting beat-level or record-level segments and mapping labels into unified formats.
  Implements three neural network architectures (CNN, LSTM, and MLP) to capture complex temporal and morphological patterns.
  Evaluates model performance using metrics such as Accuracy, F1 Score, and AUC-ROC.
  Performs cross-dataset validation by testing models trained on one dataset with data from the other, highlighting domain differences.

## Repository Structure
  ### preprocess_mitbih.py
  Preprocessing routine for the MIT-BIH dataset.<br>
  Loads ECG signals and annotations.<br>
  Extracts fixed-length (360-sample) segments around each beat.<br>
  Maps raw annotation symbols into clinically relevant classes.

  ### preprocess_ptbxl.py
  Preprocessing routine for the PTB-XL dataset.
  Loads raw 12-lead ECG signals and metadata using WFDB tools.<br>
  Aggregates diagnostic codes into broader diagnostic classes.<br>
  Encodes the aggregated labels into numerical values.

  ### model_training.py
  Training script for deep learning models (CNN, LSTM, MLP).<br>
  Contains model definitions and training routines.<br>
  Uses the preprocessed datasets to train models with Adam optimizer and Cross-Entropy Loss.

  ### model_evaluation.py
  Evaluation script for the trained models.<br>
  Computes performance metrics (Accuracy, F1 Score, AUC-ROC) and confusion matrices.<br>
  Provides saliency map visualizations for model interpretability.

  ## cross_dataset_validation.py
  Cross-dataset evaluation script.<br>
  Loads a pretrained model from one dataset and tests it on the other after appropriate preprocessing adjustments (e.g. channel replication, fixed-length segmentation, binary label mapping).

## Requirements
    Python 3.7+
    PyTorch
    NumPy
    Pandas
    scikit-learn
    WFDB (for reading ECG data)
    Matplotlib

## Usage
  ### Preprocessing:
  Run preprocess_mitbih.py to generate mitbih_signals.npy and mitbih_labels.npy.<br>
  Run preprocess_ptbxl.py to generate ptbxl_signals.npy and ptbxl_labels.npy.

  ### Training:
  Edit model_training.py to choose the dataset (mitbih or ptbxl) and the desired model type (cnn, lstm, or mlp).<br>
  Run model_training.py to train the model. The trained model weights will be saved (e.g., mitbih_cnn_model.pth).

  ### Evaluation:
  Edit model_evaluation.py to select the dataset and model type for evaluation. <br>
  Run model_evaluation.py to compute metrics and visualize saliency maps.

  ### Cross-Dataset Validation:
  Edit cross_dataset_validation.py to configure the test dataset and pretrained model source.<br>
  Run cross_dataset_validation.py to evaluate cross-dataset performance.

## Results
  The evaluation scripts compare model performance with metrics such as Accuracy, F1 Score, and AUC-ROC.<br>
  Cross-dataset validation demonstrates the impact of domain differences between MIT-BIH and PTB-XL, highlighting the need for further domain adaptation and advanced preprocessing techniques.

## References
  Project was inspired by reasearch studies:
  
  Katal, N., Gupta, S., Verma, P., & Sharma, B. (2020). Deep-learning-based arrhythmia detection using ECG signals: A comparative study and performance evaluation. arXiv preprint arXiv:2004.13701. (https://www.researchgate.net/publication/376259322_Deep-Learning-Based_Arrhythmia_Detection_Using_ECG_Signals_A_Comparative_Study_and_Performance_Evaluation) <br>
  Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2023). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. Diagnostics, 13(24), 3605. (https://www.researchgate.net/publication/344215052_Deep_Learning_for_ECG_Analysis_Benchmarks_and_Insights_from_PTB-XL)

## License
  This project is licensed under the MIT License.
