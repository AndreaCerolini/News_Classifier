# News_Classifier

This repository contains all the dataset that needs to be preprocesssed in order to be used to train and test a neural network that classifies news articles into two categories: **Real** and **Fake**.

## Repository Structure

- **`Fake.csv and Real.csv`**
  Are the original datasets

- **`pre_processing.py`**
  The script that preprocesses the dataset, producing the data used during the training and the testing, contained in the directory preprocess_data/
  
- **`News_Classifier.py`**  
  The main script that runs the training process.  
  It loads the preprocessed data, trains the model, uses early stopping, and saves the best model at the end.
