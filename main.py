import requests
import os
from dotenv import load_dotenv
from scipy import signal
import numpy as np
from PIL import Image
from io import BytesIO


def getImageFromJson(resp):
    image_url = resp[0]['url']
    r = requests.get(image_url).content
    img = Image.open(BytesIO(r))
    return img


def getNameFromJson(resp):
    name = resp[0]['breeds'][0]['name']
    return name


def imageProcessing(imageNp, mask):
    imageNew = np.zeros((imageNp.shape[0]+2, imageNp.shape[1]+2, imageNp.shape[2]), dtype=np.uint8)
    imageNew[1:-1, 1:-1, :] = imageNp[:, :, :]
    step = (imageNew.shape[0]-3) // 9
    stepI = 1

    for y in range(1, imageNew.shape[0]-1):
        if (y > step * stepI):
            stepI += 1
        for x in range(1, imageNew.shape[1]-1):
            for i in range(imageNew.shape[2]):
                imageNew[y, x, i] = np.sum(imageNew[y-1:y+2, x-1:x+2, i] * mask)
    image = np.zeros((imageNp.shape[0], imageNp.shape[1], imageNp.shape[2]), dtype=np.uint8)
    image[:, :, :] = imageNew[1:-1, 1:-1, :]
    return image


def imageProcessingScipy(imageNp, mask):
    imageNew = np.zeros_like(imageNp)
    for color in range(imageNew.shape[2]):
        imageNew[:, :, color] = signal.convolve2d(imageNp[:, :, color], mask, mode='same', boundary='wrap')
    return imageNew


mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])


def getRequests():

    API_KEY = os.getenv('API_KEY')
    IMAGE_SIZE = os.getenv('IMAGE_SIZE')
    payload = {'limit': '1', 'has_breeds': '1', 'size': IMAGE_SIZE, 'api_key': API_KEY}
    url = os.getenv('URL')
    r = requests.get(url, payload)
    if r.status_code != 200:
        print("Error download image: " + str(r.status_code))
    return r


load_dotenv()
PACKAGE_IMAGE = os.getenv('PACKAGE_IMAGE')

r = getRequests()
resp = r.json()
r_text = r.text

name_cat = getNameFromJson(resp)
print(name_cat)

image = getImageFromJson(resp)
image.save(PACKAGE_IMAGE + name_cat + "_normal.jpg")

img_np = np.array(image, dtype=np.uint8)
print("size image:", img_np.shape[0], ", ", img_np.shape[1])

print("Scipy procesing start:")
img_np_new_scipy = imageProcessingScipy(img_np, mask)
imageNewScipy = Image.fromarray(img_np_new_scipy.astype('uint8'), 'RGB')
imageNewScipy.save(PACKAGE_IMAGE + name_cat + "_scipy_0.jpg")

print("Manual procesing start:")
img_np_new = imageProcessing(img_np, mask)
imageNew = Image.fromarray(img_np_new.astype('uint8'), 'RGB')
imageNew.save(PACKAGE_IMAGE+name_cat+"_manual_0.jpg")

print("Finish")
