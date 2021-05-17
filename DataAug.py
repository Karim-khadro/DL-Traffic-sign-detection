
# This file contains the data augmentation part
# IT IS NOT EXECUTABLE 
# it only illustrates how the augmentation using Augmentor works
# and how the images are chosen randomly to be added to the test images 
# as well how the final testing and training images are selected

import pandas as pd
import os
import random
import Augmentor
import random



# Do the transformation on the orginal data 
# Generate 4000 image per class for KAGGLE & BTS

sc_df =pd.read_csv("csv/test/small_classes.csv")
headers = ["path", "class"]

for i in sc_df["id"]: #
    
    path = os.path.join("data/kaggle/BackUP/Train", str(i))
    count =len(os.listdir(path))
    p = Augmentor.Pipeline(path)
    p.random_color(probability=1,min_factor =1 ,max_factor=4)
    p.random_contrast(probability=0.4,min_factor= 1 ,max_factor=2)
    p.greyscale(probability=0.5)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2
                        , magnitude=5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.skew_corner(probability=0.5,magnitude=0.3)
    p.shear(probability=0.4,max_shear_left=1,max_shear_right=1)
    p.crop_random(probability=0.1, percentage_area=0.95)
    p.resize(height=224,width=224,probability=1)
    num_samples = 1500 - count
    p.sample(num_samples)
    num = 0
    for j in os.listdir(os.path.join("data/kaggle/BackUP/Train", str(i), "output")): 
        name =  "img00"+str(i)+str(num)+str(random.randrange(100, 9000000, 3))+".png"
        os.rename(os.path.join("data/kaggle/BackUP/Train", str(i),"output", j),os.path.join("data/kaggle/BackUP/Train", str(i),name))
        num+=1
    os.rmdir(os.path.join(path,"output"))


# Choose 600 image from each class to goo to testing
df1 = pd.read_csv("csv/Train_classifier_Final.csv")
headers = ["path", "class"]
newdfTest = pd.DataFrame(columns=headers)
newdfTrain = pd.DataFrame(columns=headers)
for i in range(80): 

    path = os.path.join("data/kaggle/Train", str(i))
    newdfTrain = df1[df1['class']==(i)]
    df = pd.DataFrame(columns=headers)
    df = newdfTrain.sample(n=600)
    newdfTest = newdfTest.append(df,ignore_index=True)
    newdfTest["path"] = newdfTest["path"].str.replace(str(i)+"/", "Test/")
    newdfTrain = newdfTrain[newdfTrain["path"].isin(df["path"]) == False]
    for j in df["path"]:
        name = str(j).split('/')[1]
        if os.path.isfile(os.path.join("data/kaggle/NewTest/Test",name)):
             name = "Test/"+str(i)+name
        else:
             name = "Test/"+ name
        os.rename("data/kaggle/Train/"+str(j),os.path.join("data/kaggle/NewTest",name))

newdfTest.to_csv("csv/Test_classifier.csv",index= None)
newdfTrain.to_csv("csv/Train_classifier_Final_2.csv", index  = None)



# Choose images for training and testing
df = pd.read_csv("csv/Train_classifier_Final.csv")
df1 = pd.read_csv("csv/Test_classifier_Final.csv")
headers = ["path", "class"]

Test = pd.DataFrame(columns=headers)
Train = pd.DataFrame(columns=headers)

l = []
lm = 0
for i in range(80): 

    newdfTest = pd.DataFrame(columns=headers)
    newdfTrain = pd.DataFrame(columns=headers)

    newdfTrain = df[df['class']==(i)]
    newdfTest = df1[df1['class']==(i)]
    l.append(len(df1[df1['class']==(i)]))
    if len(df1[df1['class']==(i)]) > 600:
        newdfTest = newdfTest.sample(n=600)
    newdfTrain = newdfTrain.sample(n=3000)
    
    Test= Test.append(newdfTest)
    Train=Train.append(newdfTrain)

Test.to_csv("csv/Test.csv",index  = None)
Train.to_csv("csv/Train.csv",index  = None)
