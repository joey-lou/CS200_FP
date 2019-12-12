from datainput import *
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageChops


class feature_extract:
    """class method for easy plotting and function call"""

    def __init__(self):
        self.names = {0: "image size",
                      1: "aspect ratio",
                      2: "red-channel average",
                      3: "green-channel average",
                      4: "blue-channel average",
                      5: "red-channel variance",
                      6: "green-channel variance",
                      7: "blue-channel variance",
                      8: "laplacian average",
                      9: "laplacian variance",

                      11: "red-channel quantiles",
                      12: "green-channel quantiles",
                      13: "blue-channel quantiles",
                      14: "laplacian quantiles",
                      15: "upsampled gray image"}

        self.funcs = {0: feature_extract.ft_size,
                      1: feature_extract.ft_aspect_ratio,
                      2: feature_extract.ft_r_mean,
                      3: feature_extract.ft_g_mean,
                      4: feature_extract.ft_b_mean,
                      5: feature_extract.ft_r_var,
                      6: feature_extract.ft_g_var,
                      7: feature_extract.ft_b_var,
                      8: feature_extract.ft_e_mean,
                      9: feature_extract.ft_e_var,

                      11: feature_extract.ft_r_quantile,
                      12: feature_extract.ft_g_quantile,
                      13: feature_extract.ft_b_quantile,
                      14: feature_extract.ft_e_quantile,
                      15: feature_extract.ft_shrinked_gray}

        self.labels = {0: "Airplanes", 1: "Bear", 2: "Blimp", 3: "Comet", 4: "Crab",
                       5: "Dog", 6: "Dolphin", 7: "Giraffe", 8: "Goat", 9: "Gorilla",
                       10: "Kangaroo", 11: "Killer-Whale", 12: "Leopards", 13: "Llama",
                       14: "Penguin", 15: "Porcupine", 16: "Teddy-Bear", 17: "Triceratops",
                       18: "Unicorn", 19: "Zebra"}

    # scalar features
    @staticmethod
    def ft_size(image):
        # Returns the pixel size of the image
        return image.size

    @staticmethod
    def ft_aspect_ratio(image):
        # Returns the aspect ratio of the image
        return image.shape[1] / image.shape[0]

    @staticmethod
    def ft_r_mean(image):
        # Returns the average of the red-channel pictures for the images
        return np.mean(image[:, :, 0]) if len(image.shape) == 3 else np.mean(image)

    @staticmethod
    def ft_g_mean(image):
        # the average of the green-channel pictures for the images
        return np.mean(image[:, :, 1]) if len(image.shape) == 3 else np.mean(image)

    @staticmethod
    def ft_b_mean(image):
        # the average of the blue-channel pictures for the images
        return np.mean(image[:, :, 2]) if len(image.shape) == 3 else np.mean(image)

    @staticmethod
    def ft_r_var(image):
        # the absolute variance of the red-channel for the images
        return np.var(image[:, :, 0]) if len(image.shape) == 3 else np.var(image)

    @staticmethod
    def ft_g_var(image):
        # the absolute variance of the blue-channel for the images
        return np.var(image[:, :, 1]) if len(image.shape) == 3 else np.var(image)

    @staticmethod
    def ft_b_var(image):
        # the absolute variance of the green-channel for the images
        return np.var(image[:, :, 2]) if len(image.shape) == 3 else np.var(image)

    @staticmethod
    def ft_e_mean(image):
        # obtain laplacian/edge mean
        return np.mean(image)

    @staticmethod
    def ft_e_var(image):
        # obtain laplacian/edge variance
        return np.var(image)

    # matrix features

    @staticmethod
    def ft_r_quantile(image):
        # find various qunatile values in red channel
        return feature_extract.quantiles(image[:, :, 0]) if len(image.shape) == 3 else feature_extract.quantiles(image[:, :])

    @staticmethod
    def ft_g_quantile(image):
        # find various qunatile values in green channel
        return feature_extract.quantiles(image[:, :, 1]) if len(image.shape) == 3 else feature_extract.quantiles(image[:, :])

    @staticmethod
    def ft_b_quantile(image):
        # find various qunatile values in blue channel
        return feature_extract.quantiles(image[:, :, 2]) if len(image.shape) == 3 else feature_extract.quantiles(image[:, :])

    @staticmethod
    def ft_e_quantile(image):
        # find quantile values in laplacian/edge
        return feature_extract.quantiles(image)

    @staticmethod
    def ft_r_bucket(image):
        # find portion of red pixels in buckets
        return feature_extract.partition(image[:, :, 0]) if len(image.shape) == 3 else feature_extract.partition(image[:, :])

    @staticmethod
    def ft_g_bucket(image):
        # find portion of green pixels in buckets
        return feature_extract.partition(image[:, :, 1]) if len(image.shape) == 3 else feature_extract.partition(image[:, :])

    @staticmethod
    def ft_b_bucket(image):
        # find portion of blue pixels in buckets
        return feature_extract.partition(image[:, :, 2]) if len(image.shape) == 3 else feature_extract.partition(image[:, :])

    @staticmethod
    def ft_shrinked_gray(image):
        # shrink image first and then convert to gray, return flattened image array
        image = feature_extract.resize_image(image)
        return feature_extract.convert_grey(image).flatten()

    @staticmethod
    def ft_get_edge(image):
        image = feature_extract.convert_grey(image)
        return cv2.Laplacian(image, cv2.CV_64F, ksize=11)

    # helper functions
    @staticmethod
    def quantiles(image_single_color, divide=np.arange(0, 1.1, 0.1)):
        return np.quantile(image_single_color.flatten(), divide)

    @staticmethod
    def partition(image_single_color, parts=10):
        # partition O(nlogn), returns portion of pixels fall into each partition
        image = image_single_color.flatten()
        part_size = 255 // parts
        n = len(image)
        res = []
        image.sort()
        count = 0
        i = 0
        current = part_size
        while i < n:
            if image[i] < current:
                count += 1
                i += 1
            else:
                current += part_size
                res = np.r_[res, count / n]
                count = 0
        # left-over pixels ignored to avoid colinearity
        return res

    @staticmethod
    def resize_image(image, xsize=20, ysize=15):
        # resize to 4x3 images
        return cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_AREA)

    @staticmethod
    def convert_grey(image):
        # convert rgb image to gray
        if len(image.shape) == 2:
            return image
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    @staticmethod
    def trim(image):
        # trim empty background spaces
        image = Image.fromarray(image)
        bg1 = Image.new(image.mode, image.size, image.getpixel((0, 0)))

        diff1 = ImageChops.difference(image, bg1)
        diff1 = ImageChops.add(diff1, diff1, 2.0, -100)
        bbox1 = diff1.getbbox()
        if bbox1:
            image = image.crop(bbox1)

        bg2 = Image.new(image.mode, image.size, image.getpixel(
            (image.width - 1, image.height - 1)))
        diff2 = ImageChops.difference(image, bg2)
        diff2 = ImageChops.add(diff2, diff2, 2.0, -100)
        bbox2 = diff2.getbbox()
        if bbox2:
            image = image.crop(bbox2)

        return np.array(image)

    # intermediate functions
    @staticmethod
    def trim_all(images):
        # convert images series
        return images.apply(feature_extract.trim)

    @staticmethod
    def edge_all(images):
        # convert images seris
        return images.apply(feature_extract.ft_get_edge)


def feature_frame(df):
    # input original training_data set
    FE = feature_extract()
    images = df.Pictures
    # trim all images
    print('Trim all images..')
    images = FE.trim_all(images)
    df_X = pd.DataFrame()

    for i in range(2):
        print('Processing..', FE.names[i])
        df_X[FE.names[i]] = images.apply(FE.funcs[i])
    # convert all images to same size first 400x300
    images = pd.Series(images.apply(lambda x: FE.resize_image(x, 400, 300)))
    laplacian = FE.edge_all(images)
    # add all scalar features
    for i in range(2, 10):
        print('Processing..', FE.names[i])
        if i in [8, 9]:
            df_X[FE.names[i]] = laplacian.apply(FE.funcs[i])
        else:
            df_X[FE.names[i]] = images.apply(FE.funcs[i])

    for j in range(11, 15):
        print('Processing..', FE.names[j])
        if j == 14:
            temp = laplacian.apply(FE.funcs[j]).tolist()
        else:
            temp = images.apply(FE.funcs[j]).tolist()
        temp = pd.DataFrame(
            temp, columns=[FE.names[j] + '_' + str(x) for x in np.arange(len(temp[0]))])
        df_X = pd.concat([df_X, temp], axis=1)
    if 'Encoding' in df.columns:
        df_X['Encoding'] = df.Encoding
    return df_X


if __name__ == "__main__":
    val_path = './20_Validation/'
    # try not to printout train_data['Pictures'] directly, takes a while
    test_data = read_test_data(val_path)
    temp = feature_frame(test_data.iloc[:10, ])
    print(temp)
