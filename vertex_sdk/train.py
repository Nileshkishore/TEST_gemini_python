#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load data
data = pd.read_csv("gs://vertex_learning/IRIS.csv")

# Prepare features and labels
array = data.values
X = array[:, 0:4]
y = array[:, 4]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Train the SVC model
svn = SVC()
svn.fit(X_train, y_train)

# Make predictions and calculate accuracy
predictions = svn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
logging.info("Model accuracy: {:.2f}%".format(accuracy * 100))

# Save the model
model_file_name = "model.pkl"
with open(model_file_name, 'wb') as model_file:
    pickle.dump(svn, model_file)

# Upload the model to the specified AIP_MODEL_DIR
# This environment variable is automatically set by Vertex AI
output_model_path = os.path.join(os.environ["AIP_MODEL_DIR"], model_file_name)
with open(model_file_name, 'rb') as model_file:
    # Use Google Cloud Storage client to upload the model
    storage_client = storage.Client()
    bucket_name = "vertex_custom_learning"  # Change if your bucket name is different
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    
    # Upload the model file to the bucket
    blob.upload_from_file(model_file)
    logging.info("Model exported to: {}".format(output_model_path))
