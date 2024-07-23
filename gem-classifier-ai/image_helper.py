import os 
import cv2
import numpy as np

img_w, img_h = 220, 220    # width and height of image

def edge_and_cut(img):
    try:
        edges = cv2.Canny(img, img_w, img_h)            
        
        if(np.count_nonzero(edges)>edges.size/10000):           
            pts = np.argwhere(edges>0)
            y1,x1 = pts.min(axis=0)
            y2,x2 = pts.max(axis=0)
            
            new_img = img[y1:y2, x1:x2]           # crop the region
            new_img = cv2.resize(new_img,(img_w, img_h))  # Convert back
        else:
            new_img = cv2.resize(img,(img_w, img_h))
    
    except Exception as e:
        print(e)
        new_img = cv2.resize(img,(img_w, img_h))
    
    return new_img
    

def prepare_image(_imgPath):
    image = None
    try:
        image = cv2.imread(_imgPath)
        image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5))) # resize the image (images are different sizes)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts an image from BGR color space to RGB
    except Exception as e:
        print(e)
    return image

def read_image_labels(_dir):
    Images, Labels = [], []
    for root, dirs, files in os.walk(_dir):
        f = os.path.basename(root)  # get class name - Amethyst, Onyx, etc       
        for file in files:
            Labels.append(f)
            # try:
            #     image = cv2.imread(root+'/'+file)              # read the image (OpenCV)
            #     image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5)))       # resize the image (images are different sizes)
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts an image from BGR color space to RGB
            #     Images.append(image)
            # except Exception as e:
            #     print(e)
            image = prepare_image(root+'/'+file)
            Images.append(image)
    Images = np.array(Images)
    return (Images, Labels)

def crop_images(Imgs):
    CroppedImages = np.ndarray(shape=(len(Imgs), img_w, img_h, 3), dtype=np.int_)

    ind = 0
    for im in Imgs: 
        x = edge_and_cut(im)
        CroppedImages[ind] = x
        ind += 1

    return CroppedImages
