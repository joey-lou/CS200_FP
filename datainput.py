# data input

from skimage import io
import os
import pandas as pd


def read_train_data(file_path):
    # Fill this function out, should return a dataframe with picture object, and correct encoding
    filemap = {0: "Airplanes", 1: "Bear", 2: "Blimp", 3: "Comet", 4: "Crab",
               5: "Dog", 6: "Dolphin", 7: "Giraffe", 8: "Goat", 9: "Gorilla",
               10: "Kangaroo", 11: "Killer-Whale", 12: "Leopards", 13: "Llama",
               14: "Penguin", 15: "Porcupine", 16: "Teddy-Bear", 17: "Triceratops",
               18: "Unicorn", 19: "Zebra"}

    image_array = []
    encoding_array = []
    for encoding in range(20):
        current_folder = os.path.join(file_path, filemap[encoding])
        for file in os.listdir(current_folder):
            if file.startswith('.'):
                continue
            image_array.append(io.imread(os.path.join(current_folder, file)))
            encoding_array.append(encoding)
    res = pd.DataFrame()
    res['Pictures'] = image_array
    res['Encoding'] = encoding_array
    return res


def read_test_data(file_path):
    image_array = []

    for file in os.listdir(file_path):
        if file.startswith('.'):
            continue
        image_array.append(io.imread(os.path.join(file_path, file)))
    res = pd.DataFrame()
    res['Pictures'] = image_array
    return res
