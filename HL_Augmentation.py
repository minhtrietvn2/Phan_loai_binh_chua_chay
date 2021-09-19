
#
import numpy
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
#from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

#
count = 0
folder_data='C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Data'
folder_temp='C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/temp'
path_2_train = 'C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/train'
path_2_test  = 'C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/test'
path_2_valid = 'C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/valid'

for cnt in range(122):
    #img = load_img(folder_data+'/'+'VN2_collected ('+'{}).jpg'.format(cnt+1))
    img = cv2.imread(folder_data+'/'+'VN2_collected ('+'{}).jpg'.format(cnt+1))
    (h, w, d) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite('C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/cv2'+'/temp('+'{}).jpg'.format(cnt+1),rotated)
    img = load_img('C:/Users/minht/Desktop/Project/Data_binh_chua_chay/Augmentation/cv2' +'/temp('+'{}).jpg'.format(cnt+1))
    img = img_to_array(img)
    data = expand_dims(img,0)

    Shift = ImageDataGenerator(width_shift_range=[-35,35])
    Flip = ImageDataGenerator(horizontal_flip=True,vertical_flip=False)
    Rotate = ImageDataGenerator(rotation_range=15)
    Brightness = ImageDataGenerator(brightness_range=(0.4,1.5))

    gen_s = Shift.flow(data,batch_size=1)
    gen_f = Flip.flow(data, batch_size=1)
    gen_r = Rotate.flow(data,batch_size=1)
    gen_b = Brightness.flow(data,batch_size=1)

    for i in range(2):

        myBacths= gen_s.next()
        myBacthf = gen_f.next()
        myBacthr= gen_r.next()
        myBacthb= gen_b.next()

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacths[0].astype('uint8')
        save_img(os.path.join(folder_temp, img_name), image)

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacthf[0].astype('uint8')
        save_img(os.path.join(folder_temp, img_name), image)

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacthr[0].astype('uint8')
        save_img(os.path.join(folder_temp, img_name), image)

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacthb[0].astype('uint8')
        save_img(os.path.join(folder_temp, img_name), image)


print('Da tao xong Augmentation, tien hang phan chia:')
ik = 0
for cnt2 in range(count):
    image_2 = load_img(folder_temp+'/temp_{}.jpg'.format(cnt2))
    cnt3 = str(ik)
    ik+=1
    img_name_2 = 'VN2_'+cnt3.zfill(4)+'.jpg'
    if      ik < 0.6*count:
        save_img(os.path.join(path_2_train, img_name_2), image_2)
    elif    0.6*count <= ik < 0.8*count:
        save_img(os.path.join(path_2_test, img_name_2), image_2)
    elif    ik >=0.8*count:
        save_img(os.path.join(path_2_valid, img_name_2), image_2)

    os.remove(folder_temp+'/temp_{}.jpg'.format(cnt2))

print('Hoan thanh xong cong viec')
cv2.waitKey(0)