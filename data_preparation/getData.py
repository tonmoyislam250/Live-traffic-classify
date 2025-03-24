import glob
import os

import numpy as np
import time

from PIL import Image

import pandas as pd


def load_and_process_training_images(train_directory):
    image_paths = glob.glob(os.path.join(train_directory, '**', '*.ppm'), recursive=True)

    images = []
    labels = []

    start_time = time.time()

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.resize((32, 32))
            image = np.array(image) / 255.0
            images.append(image)

        label = int(image_path.split(os.sep)[-2])
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Loaded", len(images), "images and", len(labels), "labels in", elapsed_time, "seconds.")

    return images, labels
    
def load_ground_truth(ground_truth_path, delimiter=";"):
    ground_truth_df = pd.read_csv(ground_truth_path, delimiter=delimiter)
    
    return ground_truth_df
    
def load_testing_images(test_directory):
    test_files = sorted(os.listdir(test_directory))

    test_images = []

    start_time = time.time()

    for test_file in test_files:
        image_path = os.path.join(test_directory, test_file)

        try:
            image = Image.open(image_path)
            image = image.resize((32, 32))
            image_array = np.array(image)
            image_array = image_array.astype('float32') / 255.0
            test_images.append(image_array)
        except Exception as e:
            print(f"Skipping file {image_path}: {str(e)}")
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time

    test_images = np.array(test_images)

    print("Loaded", len(test_images), "test images in", elapsed_time, "seconds.")

    return test_images




