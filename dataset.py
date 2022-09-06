# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:03:11 2022

@author: xiwenc
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import numpy as np

import re
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader

'''
rename the image, make the piared images have the same name in the two folder

'''
# path = os.getcwd()+'/254p RGB Images/'
# files = os.listdir(path)
# for i, file in enumerate(files):
#     OldFileName = os.path.join(path, file)
#     ID =  re.findall(r"\d+",OldFileName[-10:])[0]
#     # print(ID)
#     NewFileName = os.path.join(path, str(ID)+'.jpg')
#     os.rename(OldFileName, NewFileName)



# path = os.getcwd()+'/254p Thermal Images/'
# files = os.listdir(path)
# for i, file in enumerate(files):
#     OldFileName = os.path.join(path, file)
#     ID =  re.findall(r"\d+",OldFileName[-10:])[0]
#     # print(ID)
#     NewFileName = os.path.join(path, str(ID)+'.jpg')
#     os.rename(OldFileName, NewFileName)

    
'''
prepare data for pytorch

THREE CLASESS:
    0:NN
    1:YY
    2:YN

'''

# def c(ID):
#     if (1<= ID and ID <=13700):
#             y = 0
#     elif   (13701	<= ID and ID <=14699) \
#         or (16226	<= ID and ID <=19802) \
#         or (19900	<= ID and ID <=27183) \
#         or (27515	<= ID and ID <=31294) \
#         or (31510	<= ID and ID <=33597) \
#         or (33930	<= ID and ID <=36550) \
#         or (38031	<= ID and ID <=38153) \
#         or (43084	<= ID and ID <=45279) \
#         or (51207	<= ID and ID <=52286):
            
#         y = 1
#     else:
#         y=2
        
#     print(y)

class MyDataset(Dataset):
    def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_noise = path_ir
        self.angle_array = [90, -90, 180, -180, 270, -270]
        # self.target_size = target_size
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
        self.T = transforms.Compose([
        # transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def __getitem__(self, index):
        name = os.listdir(self.path_rgb)[index]
        ID = re.findall(r"\d+",name)[0]  
        rgb = Image.open(os.path.join(self.path_rgb, name))
        ir = Image.open(os.path.join(self.path_noise , name))
        ID = int(ID)
        
        # if (1<= ID and ID <=13700):
        #     y = 0
        # elif   (13701	<= ID and ID <=14699) \
        #     or (16226	<= ID and ID <=19802) \
        #     or (19900	<= ID and ID <=27183) \
        #     or (27515	<= ID and ID <=31294) \
        #     or (31510	<= ID and ID <=33597) \
        #     or (33930	<= ID and ID <=36550) \
        #     or (38031	<= ID and ID <=38153) \
        #     or (43084	<= ID and ID <=45279) \
        #     or (51207	<= ID and ID <=52286):
                
        #     y = 1
        # else:
        #     y=2
        
        if (1<= ID and ID <=13700):
            y = 0
        elif   (13701	<= ID and ID <=14699) \
            or (15981	<= ID and ID <=19802) \
            or (19900	<= ID and ID <=27183) \
            or (27515	<= ID and ID <=31294) \
            or (31510	<= ID and ID <=33597) \
            or (33930	<= ID and ID <=36550) \
            or (38031	<= ID and ID <=38153) \
            or (41642	<= ID and ID <=45279) \
            or (51207	<= ID and ID <=52286):
                
            y = 1
        else:
            y=2
        
            
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        if self.transform is True:
            
            rgb = self.T (rgb)
            ir  = self.T (ir)
        
        return rgb, ir,y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))
    
    
    

class MyDataset_train(Dataset):  # train for cross dataset validation
    def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_noise = path_ir
        self.angle_array = [90, -90, 180, -180, 270, -270]
        # self.target_size = target_size
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
        self.T = transforms.Compose([
        # transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def __getitem__(self, index):
        name = os.listdir(self.path_rgb)[index]
        ID = re.findall(r"\d+",name)[0]  
        rgb = Image.open(os.path.join(self.path_rgb, name))
        ir = Image.open(os.path.join(self.path_noise , name))
        ID = int(ID)
        
        if (1<= ID and ID <=13700):
            y = 0
        # elif   (13701	<= ID and ID <=14699) \
        #     or (16226	<= ID and ID <=19802) \
        #     or (19900	<= ID and ID <=27183) \
        #     or (27515	<= ID and ID <=31294) \
        #     or (31510	<= ID and ID <=33597) \
        #     or (33930	<= ID and ID <=36550) \
        #     or (38031	<= ID and ID <=38153) \
        #     or (43084	<= ID and ID <=45279) \
        #     or (51207	<= ID and ID <=52286):
                
        #     y = 1
        else:
            y=1
        
        
        # i, j, h, w = transforms.RandomCrop.get_params(clean, output_size=self.target_size)
        
        # clean = TF.crop(clean, i, j, h, w)
        # noise = TF.crop(noise, i, j, h, w)
        # if self.transform is True:
        #     if random.random() > 0.5:
        #         rgb = TF.hflip(rgb)
        #         ir = TF.hflip(ir)
            
        #     if random.random() > 0.5:
        #         angle = np.random.choice(self.angle_array, 1)
        #         rgb = TF.rotate(rgb, angle)
        #         ir = TF.rotate(ir, angle)
        
        #if random.random() > 0.9:
        #    noise = clean
            
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        if self.transform is True:
            
            rgb = self.T (rgb)
            ir  = self.T (ir)
        
        return rgb, ir,y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))
    


def MyDataset_test(path_test,input_size=254, transform=False):
    
    
    pil2tensor = transforms.ToTensor()
    T = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if transform is True:
        test_dataset = datasets.ImageFolder(path_test,T)
    else:
        test_dataset = datasets.ImageFolder(path_test,pil2tensor)
    return test_dataset


# path_test = './Test'
# test_dataset = MyDataset_test(path_test,input_size=254, transform=False)
# test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    
# cout = 1
# for i,(X,Y) in enumerate(test_dataset_loader):
#     if cout ==1 :
#         print(X.shape)
#         print(Y.shape)
#         print(i)
#         break
#     print(torch.max(X))
#     print(torch.min(X))
        
        # i, j, h, w = transforms.RandomCrop.get_params(clean, output_size=self.target_size)
        
        # clean = TF.crop(clean, i, j, h, w)
        # noise = TF.crop(noise, i, j, h, w)
        # if self.transform is True:
        #     if random.random() > 0.5:
        #         rgb = TF.hflip(rgb)
        #         ir = TF.hflip(ir)
            
        #     if random.random() > 0.5:
        #         angle = np.random.choice(self.angle_array, 1)
        #         rgb = TF.rotate(rgb, angle)
        #         ir = TF.rotate(ir, angle)
        
        #if random.random() > 0.9:
        #    noise = clean
            
    
    
    
    
    
# test
# path_rgb = './254p RGB Images/'
# path_ir = './254p Thermal Images/'
# #data = np.load(path)
# train_dataset = MyDataset(path_rgb, path_ir,input_size=224)
# train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# cout = 1
# for i,(X,driver,Y) in enumerate(train_dataset_loader):
#     if cout ==1 :
#         print(X.shape)
#         print(driver.shape)
#         print(Y.shape)
#         print(i)
    # print(torch.max(driver))
    # print(torch.min(driver))
        
#         cout = cout+1
#     else:
#         pass
#         break
# import matplotlib.pyplot as plt
# tensor_image=X[0]
# plt.imshow(tensor_image.permute(1, 2, 0)  )

# tensor_image=driver[0]
# plt.imshow(tensor_image.permute(1, 2, 0)  )


# norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
# tensor_image=X[0]
# plt.imshow(norm(tensor_image).permute(1, 2, 0)  )




