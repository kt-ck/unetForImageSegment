import cv2
import os

root = "data/test/"

images_name = os.listdir(root + "input/")
masks_name = os.listdir(root+ "prediction/")

images = [cv2.imread(root + "input/" +img, 0) for img in images_name]
masks = [cv2.imread(root+ "prediction/" + mask, 0) for mask in masks_name]

result = [cv2.addWeighted(images[index],0.8, masks[index],0.2,0.4) for index in range(len(images))]

for index in range(len(result)):
    cv2.imwrite("data/test/result/" + images_name[index], result[index])
    
