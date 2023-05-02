#!/usr/bin/env python
"""
  Script to run the document classification program

  Usage: python3 main.py
"""

# import sys
import time
from functions import csv_main_file, Dict
from classification import split_csv, create_language_models, classify_all, compute_accuracy, summary_file_path, output_folder_path

# Creating train and test sets.
# Time: 0min 2s
train_set_path, test_set_path = split_csv(csv_main_file, 0.8)
#train_set_path = csv_main_file
#test_set_path = csv_main_file

# Creating language models from the train set.
# Corpuses time: 7min 46s
# Models time: 0min 0s
models = create_language_models(train_set_path)

# Classify the test half
# Time: 1min 52s
print('Classifying...')
start_time = time.time()

#classify_all(test_set_path, models)

end_time = time.time()
print('Done. Time:' + str(end_time - start_time))

# Show accuracy
# Time: 0min 0s
print('Computing accuracy...')
start_time = time.time()

print(str(compute_accuracy(test_set_path, summary_file_path)) + '%')

end_time = time.time()
print('Done. Time:' + str(end_time - start_time))
