# data input

from skimage import io
import os
import pandas as pd
import re
import numpy as np


def read_train_data(file_path):
    # return training dataset with correct encoding labeled
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
    # read test dataset with correct orderings based on file name
    image_array = []
    image_name = []
    for file in os.listdir(file_path):
        if file.startswith('.'):
            continue
        image_name.append(re.findall(r'[0-9]+', file)[0])
        image_array.append(io.imread(os.path.join(file_path, file)))
    res = pd.DataFrame()
    res['Pictures'] = image_array
    res['Order'] = image_name
    return res.iloc[np.argsort(res.Order.apply(int)), [0]]
