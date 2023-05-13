#!/usr/bin/env python
"""
  Script to run the document classification program

"""
import time
from utilities import create_fixed_csv
from functions import csv_main_file, input_folder_path, pd
from classification import split_csv, create_language_models, classify_all, compute_accuracy, summary_file_path

def test_model(train_set_path, test_set_path):
    models = create_language_models(train_set_path)
    classify_all(test_set_path, models)
    print('Acc.: ' + str(compute_accuracy(test_set_path, summary_file_path)) + '%')

def test_model_timed(train_set_path, test_set_path):
    # Corpuses time: 7 - 10min.
    # Models time: 0min 0s
    print('Creating corpuses and models...')
    start_time = time.time()
    models = create_language_models(train_set_path)
    print('* Time: ' + str(round(time.time() - start_time, 2)) + ' sec.')
    # Classify the test half
    # Time: 2 - 10min.
    print('Classifying...')
    start_time = time.time()
    classify_all(test_set_path, models)
    print('* Time: ' + str(round(time.time() - start_time, 2)) + ' sec.')
    # Show accuracy
    # Time: 0min 0s
    print('Computing accuracy...')
    start_time = time.time()
    print('Acc.: ' + str(compute_accuracy(test_set_path, summary_file_path)) + '%')
    print('* Time: ' + str(round(time.time() - start_time, 2)) + ' sec.')

# Creating train and test sets.
# Time: 0min 2s
#train_set_path, test_set_path = csv_main_file, csv_main_file # F75_train
#train_set_path, test_set_path = split_csv(csv_main_file, 0.8) # F75_train_1, F75_train_2

# Evaluation train and test sets.
evaluation_test = input_folder_path + 'F75_test_no_clase_FIXED.csv'
train_set_path, test_set_path = csv_main_file, evaluation_test

# Testing F75_train
# F75_train
# print('Primer caso')
# test_model(csv_main_file, csv_main_file)
# F75_train_1, F75_train_2
# print('Segundo caso')
# test_model(train_set_path, test_set_path)

# Create a new csv with header: News, Classification
# create_fixed_csv(evaluation_test)
test_model(train_set_path, test_set_path)
