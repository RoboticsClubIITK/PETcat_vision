import urllib.request
import cv2
import numpy as np
import os

def store_negative_images():
    # neg_image_link="http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152"
    # neg_image_link="http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02813645"
    neg_image_link="http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00433458"
    neg_image_urls = urllib.request.urlopen(neg_image_link).read().decode()

    if not os.path.exists("negative"):
        os.makedirs("negative")

    pic = 1
    
    for i in neg_image_urls.split('\n'):
        try:
            urllib.request.urlretrieve(i,"negative/"+str(pic)+'.jpg')
            img=cv2.imread("negative/"+str(pic)+'.jpg', cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite("negative/"+str(pic)+'.jpg',resized_image)
            pic+=1
        except Exception as e:
            print(str(e))

def delete_bad_ones():
    for file_type in ['negative']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path=str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))

                    picture=cv2.imread(current_image_path)
                    if ugly.shape==picture.shape and not(np.bitwise_xor(ugly,picture).any()):
                        os.remove(current_image_path)
                        print(current_image_path)
                except Exception as e:
                    print(str(e))

def create_list():
    for file_type in ['negative']:
        for img in os.listdir(file_type):
            if file_type =='negative':
                line=file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
            elif file_type == 'positive':
                line=file_type+'/'+img+'1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)

def resize_positive_image():
    positive=cv2.imread("watch.jpg")
    resized=cv2.resize(positive,(50,50))
    cv2.imwrite("watch5050.jpg",resized)
# store_negative_images()
# resize_positive_image()
create_list()
