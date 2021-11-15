import cv2 as cv
import math
import numpy as np

def sharpen(img, sigma=100):    
    blur_img = cv.GaussianBlur(img, (0, 0), sigma)
    usm = cv.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    brightness = 0
    contrast = -100 # - 減少對比度/+ 增加對比度
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def dilation(img, x=4, y=4):    
    kernel = np.ones((x,y),np.uint8) 
    dilation = cv.dilate(img, kernel, iterations = 1)
    return dilation