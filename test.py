import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform

def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    print(X1)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = 2 * (dist_sq  / variance / variance)
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

arr = np.random.rand(2, 3, 2)
print(arr)
print('sum:', np.sum(arr, axis = 2))
print('max: ', np.repeat(arr, 2, axis = 2))