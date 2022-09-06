# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:17:58 2022

@author: xiwenc
"""



import torch  
import torch.nn as nn
import numpy as np
# import PIL.Image as Image
import torchvision.transforms as transforms
# from torch import optim
# from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn import Module
#from module.feedForward import FeedForward
#from module.multiHeadAttention import MultiHeadAttention
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleList
import math
from torch.nn.modules.loss import _WeightedLoss
import torchvision.models as models




class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out




        
        



# Model from previous paper ## fusion at the beginning
class Flame_one_stream(Module):
    
    def __init__(self
                 ):
        super().__init__()
        
        
        self.IN =  nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
        )
        
        
        self.IN_both =  nn.Sequential(
            nn.Conv2d(6, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
        )
        
        
        self.residual =  nn.Conv2d(8, 8, 1, stride=2)
        
        
        self.block =  nn.Sequential(
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
            
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),            
            nn.MaxPool2d(3, stride=2,padding=1)
            
        )
        
        
        self.block2 = nn.Sequential(
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU()
            
        )
        
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.out = nn.Sequential(
                      
            nn.Linear(8, 3),
            nn.Dropout(0.5)
            )
               

    def forward(self, x1,x2,mode):
        
        
        
        if mode == 'rgb':
            x =  x1
            x = self.IN(x) 
        elif mode == 'ir':
            x = x2
            x = self.IN(x) 
        elif mode == 'both':
            x= torch.cat((x1,x2),dim=1)
            x = self.IN_both(x) 
            
        else:
            print('Please select mode: only rgb / only ir/ both')
            
        r = x
        x = self.block(x)
        r = self.residual(r)
        x = x+r
        x = self.block2(x)
        x = self.globalpool(x)
        x = x.squeeze()
        x = self.out(x)
      
       
       
      

        return x


class Flame(Module): # feature extraction
    def __init__(self
                 ):
        super().__init__()
        
        
        self.IN =  nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
        )
        
        
        self.IN_both =  nn.Sequential(
            nn.Conv2d(6, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
        )
        
        
        self.residual =  nn.Conv2d(8, 8, 1, stride=2)
        
        
        self.block =  nn.Sequential(
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),
            
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU(),            
            nn.MaxPool2d(3, stride=2,padding=1)
            
        )
        
        
        self.block2 = nn.Sequential(
            SeparableConv2d(8, 8, 3),
            nn.BatchNorm2d(8),                            
            nn.ReLU()
            
        )
        
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x1,x2,mode):
        
        
        
        if mode == 'rgb':
            x =  x1
            x = self.IN(x) 
        elif mode == 'ir':
            x = x2
            x = self.IN(x) 
        elif mode == 'both':
            x= torch.cat((x1,x2),dim=1)
            x = self.IN_both(x) 
            
        else:
            print('Please select mode: only rgb / only ir/ both')
            
        r = x
        x = self.block(x)
        r = self.residual(r)
        x = x+r
        x = self.block2(x)
        x = self.globalpool(x)
        x = x.squeeze()
        
        return x
        


## fusion at the ending
class Flame_two_stream(Module):
    
    def __init__(self
                 ):
        super().__init__()
        self.stream1 = Flame()
        self.stream2 = Flame()
        
        self.out = nn.Sequential(
                      
            nn.Linear(8*2, 3),
            nn.Dropout(0.5)
            )
        
        
    def forward(self, x1,x2,mode):
        
        x1 = self.stream1(x1,x2,'rgb')
        x2 = self.stream2(x1,x2,'ir')
        
        x = torch.cat((x1,x2),axis = 1)
        x = self.out(x)
        
        return x
        
        
        
        
        
        
    

## pre-trained VGG
def VGG16(classes_num):
    model = models.vgg16(pretrained=True) 
    for parameter in model.parameters():
        parameter.required_grad = False
    model.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(4096, 4096),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.5),
                                      nn.Linear(4096, classes_num))
    return model





## two stream vgg. Fusion at the ending
class Vgg_two_stream(Module):
    
    def __init__(self
                 ):
        super().__init__()
        self.stream1 = models.vgg16(pretrained=True) 
        for parameter in self.stream1.parameters():
            parameter.required_grad = False
        self.stream1.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(0.5),
                                          nn.Linear(4096, 4096))
        
        self.stream2 = models.vgg16(pretrained=True) 
        for parameter in self.stream2.parameters():
            parameter.required_grad = False
        self.stream2.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(0.5),
                                          nn.Linear(4096, 4096))
        
        self.out = nn.Sequential(
                      
            nn.Linear(4096*2, 3),
            nn.Dropout(0.5)
            )
        
        
    def forward(self, x1,x2,mode):
        
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)
        
        x = torch.cat((x1,x2),axis = 1)
        x = self.out(x)
        
        return x




## pre-trained Mobilenet v2
def Mobilenetv2(classes_num):
    model = models.mobilenet_v2(pretrained=True) 
    for parameter in model.parameters():
        parameter.required_grad = False
    model.classifier = nn.Sequential(nn.Dropout(0.2),
                                      nn.Linear(1280, classes_num))
    return model



## two stream Mobilenetv2 Fusion at the ending

class Mobilenetv2_two_stream(Module):
    
    def __init__(self
                 ):
        super().__init__()
        
        self.stream1 =models.mobilenet_v2(pretrained=True) 
        for parameter in self.stream1.parameters():
            parameter.required_grad = False
        self.stream1.classifier = nn.Sequential(nn.Dropout(0.2),
                                          nn.Linear(1280, 1280))
        
        self.stream2 =models.mobilenet_v2(pretrained=True) 
        for parameter in self.stream2.parameters():
            parameter.required_grad = False
        self.stream2.classifier = nn.Sequential(nn.Dropout(0.2),
                                          nn.Linear(1280, 1280))
        
        self.out = nn.Sequential(
                      
            nn.Linear(1280*2, 3),
            nn.Dropout(0.5)
            )
        
        
    def forward(self, x1,x2,mode):
        
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)
        
        x = torch.cat((x1,x2),axis = 1)
        x = self.out(x)
        
        return x






class Logistic(nn.Module):
    def __init__(self,classes_num=3):
        super(Logistic, self).__init__()
        
        self.logic = nn.Sequential( 
            nn.Linear(254*254*3, classes_num)
        )
    
    def forward(self,x):
        x = x.view(-1, 254*254*3)
        x = self.logic(x)
        return x    



class Logistic_two_stream(nn.Module):
    def __init__(self,classes_num=3):
        super().__init__()
        
        self.IN = nn.Sequential( 
            nn.Linear(254*254*3, classes_num)
        )
        
        
        self.IN_both = nn.Sequential( 
            nn.Linear(254*254*3*2, classes_num)
        )
    
        self.flatten = nn.Flatten()
    
    def forward(self, x1,x2,mode):
        
        if mode == 'rgb':
            x =  x1
            x = self.flatten(x)
            x = self.IN(x) 
        elif mode == 'ir':
            x = x2
            x = self.flatten(x)
            x = self.IN(x) 
        elif mode == 'both':
            x= torch.cat((x1,x2),dim=1)
            x = self.flatten(x)
            x = self.IN_both(x) 
            
        else:
            print('Please select mode: only rgb / only ir/ both')
                   
        return x






## pre-trained Resnet18
def Resnet18(classes_num):
    model = models.resnet18(pretrained=True) 
    for parameter in model.parameters():
        parameter.required_grad = False
    model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=classes_num))
    return model


## two stream Resnet18

class Resnet18_two_stream(Module):
    
    def __init__(self
                 ):
        super().__init__()
        
        self.stream1= models.resnet18(pretrained=True) 
        for parameter in self.stream1.parameters():
            parameter.required_grad = False
        self.stream1.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256))
        
        self.stream2= models.resnet18(pretrained=True) 
        for parameter in self.stream2.parameters():
            parameter.required_grad = False
        self.stream2.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256))
        
        self.out = nn.Sequential(
                      
            nn.Linear(512, 3),
            nn.Dropout(0.5)
            )
        
        
    def forward(self, x1,x2,mode):
        
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)
        
        x = torch.cat((x1,x2),axis = 1)
        x = self.out(x)
        
        return x



class LeNet5_one_stream(nn.Module): # input 254*254

    def __init__(self, n_classes=3):
        super().__init__()
        
        self.feature_extractor_1 = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        
        
        
        self.feature_extractor_2 = nn.Sequential(            
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        
        

        self.classifier_1 = nn.Sequential(
            nn.Linear(in_features=376320, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(in_features=376320, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x1,x2,mode):
        
        if mode == 'rgb':
            x = self.feature_extractor_1(x1)
            x = torch.flatten(x, 1)
            y = self.classifier_1(x)
        elif mode == 'ir':
            x = self.feature_extractor_1(x2)
            x = torch.flatten(x, 1)
            y = self.classifier_1(x)
        elif mode == 'both':
            x= torch.cat((x1,x2),dim=1)
            x = self.feature_extractor_2(x)
            x = torch.flatten(x, 1)
            # print(x.shape)
            y = self.classifier_2(x)
        
        
        
        return y



class LeNet5_two_stream(nn.Module): # input 254*254
    def __init__(self
                 ):
        super().__init__()
        self.stream1 = LeNet5_one_stream().feature_extractor_1
        self.stream2 = LeNet5_one_stream().feature_extractor_1
        
        self.classifier = nn.Sequential(
                      
            nn.Linear(376320*2, 84),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(in_features=84, out_features=3)
            )
        
        
    def forward(self, x1,x2,mode):
        
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1,x2),axis = 1)
        x = self.classifier(x)
        
        return x




# test
# a = torch.rand(2,3,254,254)
# b = torch.rand(2,3,254,254)
# # Flame_one_stream()(a,b,'both')
# # Flame_one_stream()(a,b,'ir')

# # Flame_two_stream()(a,b,_)
# # Flame_two_stream()(a,b,_)

# # Vgg_two_stream()(a,b,_)
# # Mobilenetv2_two_stream()(a,b,_)


# Logistic_two_stream()(a,b,'both')
# moddd = Resnet18(3)
# moddd(a)
# Resnet18_two_stream()(a,b,'both')
#LeNet5_two_stream()(a,b,'rgb')
