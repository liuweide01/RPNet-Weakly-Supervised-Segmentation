import numpy as np
import os
import cv2
image_path = '/home/wdliu/VOC/VOC2012EX/VOC2012_SEG_AUG/segmentations'
image_list = './train_aug.txt'

with open(image_list) as f:
    content = f.readlines()

content = [x.strip() for x in content]
print(len(content))
cnt = 1

for line in content:
    image_pth = os.path.join(image_path, str(line) + '.png')

    mask = cv2.imread(image_pth)
    exist_cat = np.unique(mask).tolist()

    #exist_cat = [x-91 for x in exist_cat]

    if 0 in exist_cat:
        exist_cat.remove(0)

    for index in range(21,256):
        if index in exist_cat:
            exist_cat.remove(index)

    for cat in exist_cat:
        outF = open(os.path.join('./list/full', str(cat)+'.txt'), "a")        
        outF.write(line.strip())
        outF.write("\n")
    cnt += 1
    print(cnt,'/',len(content))
