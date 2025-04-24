import requests
import os
from dotenv import load_dotenv
from scipy import signal
import numpy as np
from PIL import Image
from io import BytesIO

class ImageProcessor:
    def __init__(self,n,resp):
        self._getNameFromJson(resp)
        self._getImageFromJson(resp)
        self.__n = n
        print(n,self.__name)
                
    def getname(self):
        return self.__name
    def getimg(self):
        return self.__img
    
    def _getNameFromJson(self,resp):
        self.__name = resp['breeds'][0]['name']
        
    def _getImageFromJson(self,resp):
        image_url = resp['url']
        r = requests.get(image_url).content
        self.__img = Image.open(BytesIO(r))

    @staticmethod
    def imageProcessing(mask,imageNp):
        print(imageNp.shape)
        imageNew = np.zeros((imageNp.shape[0]+2, imageNp.shape[1]+2, imageNp.shape[2]), dtype=np.uint8)
        imageNew[1:-1, 1:-1, :] = imageNp[:, :, :]
        for y in range(1, imageNew.shape[0]-1):
            for x in range(1, imageNew.shape[1]-1):
                for i in range(imageNew.shape[2]):
                    imageNew[y, x, i] = np.sum(imageNew[y-1:y+2, x-1:x+2, i] * mask)
        image = np.zeros((imageNp.shape[0], imageNp.shape[1], imageNp.shape[2]), dtype=np.uint8)
        image[:, :, :] = imageNew[1:-1, 1:-1, :]
        return image
    @staticmethod
    def imageProcessingScipy(mask, imageNp):
        imageNew = np.zeros_like(imageNp)
        for color in range(imageNew.shape[2]):
            imageNew[:, :, color] = signal.convolve2d(imageNp[:, :, color], mask, mode='same', boundary='wrap')
        return imageNew
    
    def imgOriginalSave(self,PACKAGE_IMAGE, filename):
        self.__img.save(PACKAGE_IMAGE + str(self.__n) + "_"+ self.__name + filename)

    def getOriginalNpArray(self):
        return np.array(self.__img, dtype=np.uint8)
    

    def NpArraySave(self,PACKAGE_IMAGE,np,filename):
         image = Image.fromarray(np.astype('uint8'), 'RGB')
         image.save(PACKAGE_IMAGE + str(self.__n) + "_"+ self.__name + filename)

        
    @staticmethod
    def makelistImageProcessor(resp):
        listimages = []
        for i in range(len(resp)):
            listimages.append(ImageProcessor(i,resp[i]))
        return listimages
    


if not os.path.exists(".env"):
    print("[Error] File .env not found. Plese create! (API_KEY, URL, IMAGE_SIZE, PACKAGE_IMAGE)")
    exit(1)

    

load_dotenv()
PACKAGE_IMAGE = os.getenv('PACKAGE_IMAGE')
if not os.path.exists(PACKAGE_IMAGE):
    os.makedirs(PACKAGE_IMAGE)
PACKAGE_IMAGE = os.path.join(PACKAGE_IMAGE, "")
API_KEY = os.getenv('API_KEY')
IMAGE_SIZE = os.getenv('IMAGE_SIZE')
payload = {'limit': '3', 'has_breeds': '1', 'size': IMAGE_SIZE, 'api_key': API_KEY}
url = os.getenv('URL')
r = requests.get(url, payload)
if r.status_code != 200:
    print("Error download image: " + str(r.status_code))


resp = r.json()

mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

for obj in ImageProcessor.makelistImageProcessor(resp):

    obj.imgOriginalSave(PACKAGE_IMAGE, "_original.jpg")

    

    print("Scipy procesing start:")
    img_np_new_scipy = ImageProcessor.imageProcessingScipy(mask, obj.getOriginalNpArray())
   
    obj.NpArraySave(PACKAGE_IMAGE,img_np_new_scipy,"_scipy.jpg")

    
    print("Manual procesing start:")
    img_np_new = ImageProcessor.imageProcessing(mask,obj.getOriginalNpArray())
    obj.NpArraySave(PACKAGE_IMAGE,img_np_new,"_processed.jpg")
    
    print("Finish")
