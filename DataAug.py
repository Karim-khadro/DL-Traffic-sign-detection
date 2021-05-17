#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from numpy import asarray
import matplotlib.pyplot as plt 
import pandas as pd
import os
import random
from torchvision.utils import save_image
import copy


# In[16]:


# KAGGLE
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing vehicle over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'vehicle > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing vehicle > 3.5 tons' }




train_img_path = "data/backup/Train"

folders = os.listdir(train_img_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_img_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    
    
# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.title("Kaggle Data per class")
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()

print(sorted_pairs)


# In[ ]:



classes = {
    0: "Bumpy road ",
    1: "Road hump ahead",
    2: "Slippery road ",
    3: "Dangerous curve left ",
    4: "Dangerous curve right",
    5: "Double curve left then right",
    6: "Double curve right then left ",
    7: "Children crossing ",
    8: "Bicycles",
    9: "Wild animals crossing",
    10: "Road work",
    11: "Traffic signals ",
    12: "Level crossing with barriers ahead",
    13: "General caution ",
    14: "Road narrows on both sides",
    15: "Road narrows on left side",
    16: "Road narrows on right side",
    17: "Crossroads with a minor road",
    18: "Crossroads with priority to the right",
    19: "Give Way \/ Yield",
    20: "Give way to oncoming traffic",
    21: "Stop",
    22: "No entry",
    23: "No bicycles",
    24: "Maximum weight",
    25: "No heavy goods vehicles",
    26: "Maximum width 2.1 m",
    27: "Maximum height",
    28: "No vehicles",
    29: "No left turn",
    30: "No right turn",
    31: "No overtaking",
    32: "Speed limit (70km\/h)",
    33: "Bicycles and pedestrians only",
    34: "Straight ahead only",
    35: "Proceed left only",
    36: "Proceed straight or turn right only",
    37: "Roundabout",
    38: "Bicycles only",
    39: "Road for bicycles and pedestrians only",
    40: "No parking or waiting",
    41: "No stopping",
    42: "No parking or waiting form 1 to 15 of the mounth",
    43: "No parking or waiting form 16 to end of the mounth",
    44: "Priority over oncoming vehicles",
    45: "Parking",
    46: "Handicapped parking",
    47: "Car only parking",
    48: "Truck only parking",
    49: "Bus only parking",
    50: "Sidewalk parking",
    51: "Residential area \/ living street",
    52: "End of residential area \/ living street",
    53: "One-way street",
    54: "No through road",
    55: "End of road work",
    56: "Pedestrian crossing",
    57: "Bicycles crossing",
    58: "Parking announcement",
    59: "Road hump",
    60: "End of priority road",
    61: "Priority road"
}
train_img_path = "data\B_TS\Training"

folders = os.listdir(train_img_path)

train_number = []
class_num = []
i = 0
for folder in folders:
    train_files = os.listdir(train_img_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[i])
    i = i+1


# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.title("BTS Data per class -TRAINING-")
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()


# In[2]:


class RoadSignDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, transform = None ):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.df = df
        
    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.df)
        
    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        img = torchvision.datasets.folder.pil_loader(os.path.join(self.root_dir, self.df["path"][index]))
        if self.transform != None:
            img = self.transform(img)
        label = self.df["class"][index]
        return img,label
    


# In[ ]:


df = pd.read_csv("data/B_TS/AllTraining.csv")
data = [df["Path"],df["ClassId"]]
headers = ["path", "class"]
trainDF = pd.concat(data, axis=1, keys=headers)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# TEST CAN BE DELETED
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = Image.open('data/B_TS/AnnoTraining/0.png')

# Create figure and axes
fig, ax = plt.subplots()
# Display the image
ax.imshow(im)
# Create a Rectangle patch
rect = patches.Rectangle((1346.82, 246.76), 1582.12-1346.82, 484.41-246.76, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
plt.show()


# In[ ]:


# DONE FOR TRAINING
# DATA processing : transform all jp2 to png 
from openjpeg import decode

df = pd.read_csv("csv/BTSD_training_GT.csv")
dfNew = copy.deepcopy(df) 
for i in range(len(df["path"])):
    #Decode jp2
    with open(os.path.join('data/B_TS/Annotation',(df['path'][i])), 'rb') as f:
        # Returns a numpy array
        arr = decode(f)
        image = Image.fromarray(arr, 'RGB')
        name = "AnnoTraining/"+str(i)+".png"
        image.save(os.path.join('data/B_TS',name))
    dfNew['path'][i] = dfNew['path'][i].replace(dfNew['path'][i],name)

dfNew.to_csv("csv/yolo.csv",index=None)


# In[4]:


# DONE FOR TESTING
# DATA processing : transform all jp2 to png 
from openjpeg import decode

df = pd.read_csv("csv/BTSD_testing_GT.csv")
dfNew = copy.deepcopy(df) 
for i in range(len(df["path"])):
    #Decode jp2
    with open(os.path.join('data/B_TS/Annotation',(df['path'][i])), 'rb') as f:
        # Returns a numpy array
        arr = decode(f)
        image = Image.fromarray(arr, 'RGB')
        name = "AnnoTesting/"+str(i)+"00.png"
        image.save(os.path.join('data/B_TS',name))
    dfNew['path'][i] = dfNew['path'][i].replace(dfNew['path'][i],name)

dfNew.to_csv("csv/yoloTest.csv",index=None)


# In[ ]:


# DONE FOR TRAINING
# Make images for yolo use 

def BndBox2YoloLine(df,width, height):
    xmin = df['x1']
    xmax = df['x2']
    ymin = df['y1']
    ymax = df['y2']

    xcen = float((xmin + xmax)) / 2 / width
    ycen = float((ymin + ymax)) / 2 / height

    w = float((xmax - xmin)) / width
    h = float((ymax - ymin)) / height

    boxName = str(df['path']).split("/")[1]
    boxName = boxName.split(".")[0]

    return 0, xcen, ycen, w, h,boxName#,xmin,xmax,ymin,ymax


df = pd.read_csv("csv/yolo.csv")
for i in range(len(df['path'])):
    image = Image.open(os.path.join("data/B_TS",(df['path'][i]))).convert("RGBA")
    width, height = image.size
    c,x,y,w,h,name = BndBox2YoloLine(df.iloc[[i]],width, height)
    fname = name
    file1 = open(os.path.join("data/B_TS/AnnoTraining",fname+".txt"),"w")
    # \n is placed to indicate EOL (End of Line)
    file1.write(f"{c} {x} {y} {w} {h}")
    file1.close()


# In[ ]:


# YOLO datset for testing and to add for calssifier data
from openjpeg import decode
# cols_to_use = ["path","x1","y1","x2","y2","classId","superclassId","poleId","numberOnPole","cameraNumber","frameNumber","classLabel"] 
# df = pd.read_csv("csv/BTSD_testing_GT.csv", sep=";",usecols= cols_to_use)
# df.to_csv("csv/BTSD_testing_GT.csv", index=None)

df = pd.read_csv("csv/BTSD_testing_GT.csv")
z = df['classLabel'].value_counts(ascending=True)
print(len(z))
print(z)


# In[ ]:





# In[10]:


# Decode BTS classifier trainning & testing images
# DONE FOR TRAINING
df = pd.read_csv("csv/AllTesting.csv")
dfNew = copy.deepcopy(df) 

for i in range(len(df["path"])):
    path = os.path.join('data/B_TS/Testing',(df['path'][i]))
    if(os.path.isfile(path)):
        with open(path, 'rb') as f:
            image = Image.open(f)
            n = dfNew['path'][i].split(".ppm")[0]
            n = n.split("/")[1]+str(i)+"0"
            name = "ClassifierTesting/" + n+".png"
            image.save(os.path.join('data/B_TS',name))
    #     os.remove('data/B_TS/Training/' + df['path'][i])
        dfNew['path'][i] = dfNew['path'][i].replace(dfNew['path'][i],name)



dfNew.to_csv("csv/NewwAllTesting.csv",index=None)


# In[ ]:


# Create the classifier training csv file
# DONE FOR TRAINING
path = "data/kaggle/Train"
headers = ["path", "class"]
total = 0
df = pd.DataFrame(columns=headers)
for f in os.listdir(path):
    for im in  os.listdir(os.path.join(path,f)):
        total += 1
        s = f+"/"+im
        row = {'path': s, 'class': f}
        df = df.append(row, ignore_index=True)
df.to_csv("csv/ClassifierTraining.csv",index=None)


# In[8]:


classes = {
  0:"Speed limit (20km/h)"  ,
  1:"Speed limit (30km/h) "  ,
  2:"Speed limit (50km/h) "  ,
  3:"Speed limit (60km/h) "  ,
  4:"Speed limit (70km/h) "  ,
  5:"Speed limit (80km/h) "  ,
  6:"End of speed limit (80km/h) "  ,
  7:"Speed limit (100km/h) "  ,
  8:"Speed limit (120km/h) "  ,
  9:"No passing "  ,
  10:"No passing vehicle over 3.5 tons "  ,
  11:"Right-of-way at intersection "  ,
  12:"Priority road "  ,
  13:"Yield \"Give priority\""  ,
  14:"Stop "  ,
  15:"No vehicles "  ,
  16:"vehicle > 3.5 tons prohibited "  ,
  17:"No entry "  ,
  18:"General caution "  ,
  19:"Dangerous curve left "  ,
  20:"Dangerous curve right "  ,
  21:"Double curve "  ,
  22:"Bumpy road "  ,
  23:"Slippery road "  ,
  24:"Road narrows on the right "  ,
  25:"Road work "  ,
  26:"Traffic signals "  ,
  27:"Pedestrians "  ,
  28:"Children crossing "  ,
  29:"Bicycles "  ,
  30:"Beware of ice/snow"  ,
  31:"Wild animals crossing "  ,
  32:"End speed + passing limits "  ,
  33:"Turn right ahead "  ,
  34:"Turn left ahead "  ,
  35:"Ahead only "  ,
  36:"Go straight or right "  ,
  37:"Go straight or left "  ,
  38:"Keep right "  ,
  39:"Keep left "  ,
  40:"Roundabout mandatory "  ,
  41:"End of no passing "  ,
  42:"End no passing vehicle > 3.5 tons "  ,
  43:"Bumpy road",
  44:"Double curve right then left"  ,
    45:"End of priority road"  ,
  46:"Level crossing with barriers ahead"  ,
  47:"Road narrows on both sides"  ,
  48:"Road narrows on left side"  ,
  49:"Crossroads with priority to the right"  ,
  50:"Give way to oncoming traffic"  ,
  51:"No bicycles"  ,
  52:"Maximum weight"  ,
  53:"Maximum width 2.1 m"  ,
  54:"Maximum height"  ,
  55:"No left turn"  ,
  56:"No right turn"  ,
  57:"Bicycles and pedestrians only"  ,
  58:"Proceed left only"  ,
  59:"Bicycles only"  ,
  60:"Road for bicycles and pedestrians only"  ,
  61:"No parking or waiting"  ,
  62:"No stopping"  ,
  63:"No parking or waiting form 1 to 15 of the mounth"  ,
  64:"No parking or waiting form 16 to end of the mounth"  ,
  65:"Priority over oncoming vehicles"  ,
  66:"Parking"  ,
  67:"Handicapped parking"  ,
  68:"Car only parking"  ,
  69:"Truck only parking"  ,
  70:"Bus only parking"  ,
  71:"Sidewalk parking"  ,
  72:"Residential area / living street"  ,
  73:"End of residential area / living street"  ,
  74:"One-way street"  ,
  75:"No through road"  ,
  76:"End of road work"  ,
  77:"Bicycles crossing"  ,
  78:"Parking announcement"  ,
  79:"Road hump"  

}


train_img_path = "data\kaggle\BackUP\Train"

folders = os.listdir(train_img_path)

train_number = []
class_num = []

headers = ["name", "id", "number"]
small_class_df = pd.DataFrame(columns=headers)

for folder in folders:
    train_files = os.listdir(train_img_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    if len(train_files) < 1500:
        class_name = classes[int(folder)]
        class_id = int(folder)
        class_count = len(train_files)
        row = {'name': class_name, 'id': class_id, 'number': class_count}
        small_class_df = small_class_df.append(row, ignore_index = True)

small_class_df.to_csv("csv/test/small_classes.csv",index=None)

# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.title("Classifier training data per class")
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()


# In[ ]:





# In[ ]:


# GOAL 1100 images by class
transformsList = []
# OK
# transformsList.append(transforms.Compose([transforms.Resize((200,200)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5,hue=0.5),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# OK
# transformsList.append(transforms.Compose([transforms.Resize((180,180)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=0, contrast=(1.5,3.5), saturation=0, hue=0),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# # OK
# transformsList.append(transforms.Compose([transforms.Resize((224,224)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=3, contrast=0, saturation=(4,7), hue=0),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# # OK
# transformsList.append(transforms.Compose([transforms.Resize((100,100)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=7, contrast=0, saturation=0, hue=(-0.40,0.05)),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))



# transformsList.append(transforms.Compose([transforms.Resize((80,80)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=4, contrast=1.2, saturation=1.8, hue=0),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))


# transformsList.append(transforms.Compose([transforms.Resize((224,224)),
#                                 transforms.ToTensor(),
#                                 transforms.ColorJitter(brightness=8, contrast=1, saturation=2.5, hue=0),
#                                 transforms.Grayscale(num_output_channels=3),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# # transformsList.append(transforms.Compose([transforms.Resize((224,224)),
# #                                 transforms.ToTensor(),
# #                                 transforms.Pad(random.randint(0, 10), fill=0, padding_mode='constant'),
# #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
                      
    
# transformsList.append(transforms.Compose([transforms.Resize((224,224)),
#                                 transforms.transforms.RandomRotation(15),
#                                 transforms.ColorJitter(brightness=10, contrast=1, saturation=2.5, hue=0),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
                      


# In[ ]:





# In[12]:


# DONE
# Do the transformation on the orginal data 
# Generate 4000 image per class for KAGGLE & BTS
# 200 images are for testing

import Augmentor
import random

sc_df =pd.read_csv("csv/test/small_classes.csv")
headers = ["path", "class"]

for i in sc_df["id"]: # 
    
    path = os.path.join("data/kaggle/BackUP/Train", str(i)) #"data/kaggle/newTraining"
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


# In[11]:


# Remake the csv files for TEST & TRAIN

path = "data/kaggle/Train"
headers = ["path", "class"]
df = pd.DataFrame(columns=headers)
for f in os.listdir(path):
    print(int(f))
    for im in  os.listdir(os.path.join(path,f)):
        s = f+"/"+im
        row = {'path': s, 'class': f}
        df = df.append(row, ignore_index=True)
        
df.to_csv("csv/Train_classifier_Final.csv",index=None)


# In[2]:


# AFTER augmuntaion 
classes = {
  0:"Speed limit (20km/h)"  ,
  1:"Speed limit (30km/h) "  ,
  2:"Speed limit (50km/h) "  ,
  3:"Speed limit (60km/h) "  ,
  4:"Speed limit (70km/h) "  ,
  5:"Speed limit (80km/h) "  ,
  6:"End of speed limit (80km/h) "  ,
  7:"Speed limit (100km/h) "  ,
  8:"Speed limit (120km/h) "  ,
  9:"No passing "  ,
  10:"No passing vehicle over 3.5 tons "  ,
  11:"Right-of-way at intersection "  ,
  12:"Priority road "  ,
  13:"Yield \"Give priority\""  ,
  14:"Stop "  ,
  15:"No vehicles "  ,
  16:"vehicle > 3.5 tons prohibited "  ,
  17:"No entry "  ,
  18:"General caution "  ,
  19:"Dangerous curve left "  ,
  20:"Dangerous curve right "  ,
  21:"Double curve "  ,
  22:"Bumpy road "  ,
  23:"Slippery road "  ,
  24:"Road narrows on the right "  ,
  25:"Road work "  ,
  26:"Traffic signals "  ,
  27:"Pedestrians "  ,
  28:"Children crossing "  ,
  29:"Bicycles "  ,
  30:"Beware of ice/snow"  ,
  31:"Wild animals crossing "  ,
  32:"End speed + passing limits "  ,
  33:"Turn right ahead "  ,
  34:"Turn left ahead "  ,
  35:"Ahead only "  ,
  36:"Go straight or right "  ,
  37:"Go straight or left "  ,
  38:"Keep right "  ,
  39:"Keep left "  ,
  40:"Roundabout mandatory "  ,
  41:"End of no passing "  ,
  42:"End no passing vehicle > 3.5 tons "  ,
  43:"Bumpy road",
  44:"Double curve right then left"  ,
  45:"End of priority road"  ,
  46:"Level crossing with barriers ahead"  ,
  47:"Road narrows on both sides"  ,
  48:"Road narrows on left side"  ,
  49:"Crossroads with priority to the right"  ,
  50:"Give way to oncoming traffic"  ,
  51:"No bicycles"  ,
  52:"Maximum weight"  ,
  53:"Maximum width 2.1 m"  ,
  54:"Maximum height"  ,
  55:"No left turn"  ,
  56:"No right turn"  ,
  57:"Bicycles and pedestrians only"  ,
  58:"Proceed left only"  ,
  59:"Bicycles only"  ,
  60:"Road for bicycles and pedestrians only"  ,
  61:"No parking or waiting"  ,
  62:"No stopping"  ,
  63:"No parking or waiting form 1 to 15 of the mounth"  ,
  64:"No parking or waiting form 16 to end of the mounth"  ,
  65:"Priority over oncoming vehicles"  ,
  66:"Parking"  ,
  67:"Handicapped parking"  ,
  68:"Car only parking"  ,
  69:"Truck only parking"  ,
  70:"Bus only parking"  ,
  71:"Sidewalk parking"  ,
  72:"Residential area / living street"  ,
  73:"End of residential area / living street"  ,
  74:"One-way street"  ,
  75:"No through road"  ,
  76:"End of road work"  ,
  77:"Bicycles crossing"  ,
  78:"Parking announcement"  ,
  79:"Road hump"  
  
}


train_img_path = "data\kaggle\Train"

folders = os.listdir(train_img_path)

train_number = []
class_num = []

headers = ["name", "id", "number"]
small_class_df = pd.DataFrame(columns=headers)

for folder in folders:
    train_files = os.listdir(train_img_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])

# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.title("Classifier training data per class")
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()


# In[ ]:





# In[10]:


# Choose 200 image from each class to goo to testing

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


# In[12]:


# Add all testing csv togther
# DONE
headers = ["path", "class"]


df = pd.read_csv("data/kaggle/Test.csv")
data = [df["Path"],df["ClassId"]]
testDF = pd.concat(data, axis=1, keys=headers)

newdfTest= pd.read_csv("csv/Test_classifier.csv")

btsdf = pd.read_csv("csv/Test_bts.csv")
data = [btsdf["path"],btsdf["class"]]
btsdf = pd.concat(data, axis=1, keys=headers)

print(testDF)
print(newdfTest)
print(btsdf)

data = [testDF,newdfTest,btsdf]
finalTestDf = pd.concat(data, sort=False)
print(finalTestDf)
finalTestDf = finalTestDf.sample(frac = 1)
finalTestDf.to_csv("csv/Test_classifier_Final.csv", index  = None)


# In[ ]:





# In[11]:


#  Sampels to test colab
df = pd.read_csv("csv/Train_classifier_Final.csv")
df1 = pd.read_csv("csv/Test_classifier_Final.csv")
headers = ["path", "class"]

Test = pd.DataFrame(columns=headers)
Train = pd.DataFrame(columns=headers)
for i in range(80): 
    newdfTest = pd.DataFrame(columns=headers)
    newdfTrain = pd.DataFrame(columns=headers)

    newdfTrain = df[df['class']==(i)]
    newdfTest = df1[df1['class']==(i)]
    print(newdfTest)
    newdfTest = newdfTest.sample(n=8)
    newdfTrain = newdfTrain.sample(n=15)
    
    Test= Test.append(newdfTest)
    Train=Train.append(newdfTrain)

Test.to_csv("csv/test/colabtest.csv",index  = None)
Train.to_csv("csv/test/colabtrain.csv",index  = None)


# In[4]:


#  Smaller datasets 1500 train

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

Test.to_csv("csv/test/Test.csv",index  = None)
Train.to_csv("csv/test/Train.csv",index  = None)
print(min(l))
print(max(l))


# In[4]:


# Yolo test claean

df = pd.read_csv("csv/yoloTest.csv")

df = df[df['classId']!= -1]
print(df)
df.to_csv("csv/cleanYoloTest.csv",index  = None)


# In[2]:


from PIL import Image
from numpy import asarray


class RoadSignDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, transform):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.df = df
        
    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.df)
        
    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        
        x = True
        i = 1
        while x:
            try:
                img = torchvision.datasets.folder.pil_loader(os.path.join(self.root_dir, self.df["Path"][index+i%len(self.df)]))
                x =False
            except:
                i += 1
                
        if self.transform is not None:
            img = self.transform(img)
        label = self.df["ClassId"][index]
        
        return img,label
    
# transform = transforms.Compose([transforms.Resize((64,64)), 
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.385, 0.356, 0.366],
#                                                     std=[0.289, 0.284, 0.275],)])

# transform = transforms.Compose([transforms.Resize((64,64)), 
#                                 transforms.ToTensor()])
#   transform = transforms.Compose([transforms.Resize((64,64)), 
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# In[23]:


def show_images(img):
    img = img 
    npimg = img.numpy() * .5 + .5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
trainDf = pd.read_csv("data/backup/Train.csv")
mean_test = torch.tensor(0.3653)
std_test = torch.tensor(0.2976)


Train = pd.DataFrame()
for i in range(43): 
    newdfTrain = pd.DataFrame()
    newdfTrain = trainDf[trainDf['ClassId']==(i)]
    Train = newdfTrain.sample(n=1)
    
transform = transforms.Compose([transforms.Resize((32,32)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean_test, std_test)])

my_dataset = RoadSignDataset('data/backup' ,trainDf,transform=transform) # training directory
my_loader = torch.utils.data.DataLoader(my_dataset, batch_size=42, shuffle=True, num_workers=0)

dataiter = iter(my_loader)
images, labels = dataiter.next()
show_images(torchvision.utils.make_grid(images))


# In[ ]:




