# Neural_Network_Cancer_Detection

🔍 Overview
This project builds a deep learning model to detect metastatic cancer in histopathology image patches using the PatchCamelyon (PCam) dataset. PCam is a well-known benchmark that frames the clinically important task of lymph node tumor detection as a binary image classification problem. The model classifies 32x32 pixel image patches as either containing tumor tissue or not, with the goal of assisting pathologists in cancer diagnosis.

🎯 Objectives
Build a convolutional neural network (CNN) to classify histology images

Evaluate performance using AUC (Area Under the ROC Curve), the standard metric for medical classification tasks

Improve training using techniques like batch normalization, early stopping, and dropout

Submit predictions for evaluation on Kaggle

🗂️ Features
Custom PyTorch Dataset and DataLoader for .tif histopathology images

Simple CNN with batch normalization and dropout

Early stopping based on validation AUC to avoid overfitting

Data preprocessing and exploratory data analysis (EDA)

Kaggle-compatible prediction file generator

📊 Tools & Libraries
Python, PyTorch, torchvision

Pandas, NumPy, Matplotlib, Seaborn

scikit-learn (for evaluation)

tifffile (for loading .tif images)

🧪 Future Improvements
Incorporate pretrained models like ResNet or EfficientNet (transfer learning)

Apply data augmentation (e.g., rotations, stain normalization)

Use Grad-CAM or SHAP for model explainability

Explore uncertainty quantification with Monte Carlo dropout or ensembling

Build a full-slide inference pipeline from patch-level predictions
