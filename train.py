# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:16:56 2022

@author: xiwenc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:11:50 2022

@author: xiwenc


fine-tine pre-trained network

"""



import torch
import torch.nn as nn
import torchvision.models as models
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn.functional as F

import datetime
import os
import random

from models import Logistic_two_stream,Flame_one_stream,VGG16,Vgg_two_stream,Logistic,Flame_two_stream,Mobilenetv2,Mobilenetv2_two_stream,LeNet5_one_stream,LeNet5_two_stream,Resnet18,Resnet18_two_stream

import json
import argparse

from torchmetrics.functional import precision_recall,f1_score
import csv

import numpy as np
import sys


# model = models.vgg16(pretrained=True) 
# for parameter in model.parameters():
#     parameter.required_grad = False
# model.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Dropout(0.5),
#                                  nn.Linear(4096, 4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Dropout(0.5),
#                                  nn.Linear(4096, classes_num))





# split_rate = 0.8
# train_set, val_set = torch.utils.data.random_split(Dataset, [int(len(Dataset)*split_rate), len(Dataset)-int(len(Dataset)*split_rate)])

# train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)



# optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0.00)

# loss_function =nn.CrossEntropyLoss(label_smoothing=0.2)
# correct_on_train = [0]
# correct_on_test = [0]

# total_step = (int(len(Dataset)*split_rate)// BATCH_SIZE  )


def test(dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = False):
    
    #global _,_,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list
    
    correct = 0
    total = 0
    
    y_all_true = torch.tensor([]).to(DEVICE)
    y_all_pre  = torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        net.eval()
        for  (rgb, ir, y)  in dataloader:
            #print('1')
            y = y.to(DEVICE)
            y_all_true = torch.cat((y_all_true,y))
            # _, label_true = torch.max(y.data, dim=-1)
            
            
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE),ir.to(DEVICE),mode = MODE)
            else:
                if MODE=='rgb':
                    x = rgb.to(DEVICE)
                elif  MODE=='ir':
                    x = ir.to(DEVICE)
                    
                y_pre = net(x)
            
            
            _, label_index = torch.max(y_pre.data, dim=-1)
            y_all_pre = torch.cat((y_all_pre,label_index))
            
            total += label_index.shape[0]
            correct += (label_index == y).sum().item()
            
        if flag == 'test_set':
            
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            #label_index = F.one_hot(label_index, num_classes=3)
            
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))
        
    if output_flag == True:
        # print(y_all_pre)
        # print(y_all_true)
        macro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), num_classes=3,average ='macro')
        micro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), average ='micro')
        
        macro_p,macro_r = precision_recall(y_all_pre.int(), y_all_true.int(), average='macro', num_classes=3)
        micro_p,micro_r = precision_recall(y_all_pre.int(), y_all_true.int(), average='micro')
        
        with open(log_path,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [ Model_name,MODE,name_index,macro_f1.item() ,micro_f1.item()  ,macro_p.item()  ,macro_r.item()  ,micro_p.item()  ,micro_r.item()  ]
            csv_write.writerow(data_row)
        
    return correct / total


    


 
if __name__ == "__main__":
    
    # def main():
    parser = argparse.ArgumentParser(description='Run dgr experiment.')
    
    
    
    parser.add_argument('--path_rgb', type=str, default='./254p RGB Images/', help='results path')
    parser.add_argument('--path_ir', type=str, default='./254p Thermal Images/', help='results path')
    
    parser.add_argument('--index', type=int, default= 0, help='name index. any number is fine')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, 1e-4 for small subset') #1e-4
    parser.add_argument('--classes_num', type=int, default=3, help='class number')
    parser.add_argument('--subset_rate', type=float, default=0.01, help='The rate of subset')
    parser.add_argument('--trainset_rate', type=float, default=0.8, help='split data to training and test')
    parser.add_argument('--model', type=str, default='Flame_one_stream', help='VGG16 and Flame_one_stream')
    parser.add_argument('--mode', type=str, default='rgb', help='rgb/ir/both')
    

    
    parser.add_argument('--EPOCH', type=int, default=10, help='Epoch for training')
    parser.add_argument('--test_interval', type=int, default=1, help='interval to report the results')
    parser.add_argument('--log_path', type=str, default='./log/results.csv', help='results path')
    parser.add_argument('--log_loss_path', type=str, default='./log/', help='results path to store loss info')
    # parser.add_argument('--epsilon', type=float, default=25.5/255, help='PGD epsilon')
    # parser.add_argument('--inner_loop', type=int, help='inner loop')
    
    args = parser.parse_args()
    print(args)
    
    
     
    #path of data and log
    log_path =  args.log_path
    
    
    

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')
    
    classes_num = args.classes_num 
    BATCH_SIZE = args.batch_size
    LR = args.lr
    test_interval = args.test_interval
    EPOCH =args.EPOCH
    MODE = args.mode  # 'only rgb' 
    Model_name  = args.model
    name_index = args.index
    
    Transform_flag = False # sometime use it to resize the image
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')
    
    
    
    file_name =Model_name+'_'+MODE+'_'+str(args.index)
    
    # def model
    
    
    Model_custom_list = ['Flame_one_stream','Flame_two_stream','Mobilenetv2_two_stream','Vgg_two_stream','Logistic_two_stream','Resnet18_two_stream','LeNet5_one_stream','LeNet5_two_stream']
    
    if Model_name == 'Flame_one_stream':
        net = Flame_one_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'Flame_two_stream':
        net = Flame_two_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'VGG16':
        net = VGG16(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Vgg_two_stream':
        Transform_flag = True
        target_size = 224
        net = Vgg_two_stream().to(DEVICE)
        
    elif Model_name == 'Logistic':
        net= Logistic(classes_num).to(DEVICE)
        target_size = 254
        
    elif Model_name == 'Logistic_two_stream':
        net= Logistic_two_stream(classes_num).to(DEVICE)
        target_size = 254
    
   
    elif Model_name == 'Mobilenetv2':
        net = Mobilenetv2(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Mobilenetv2_two_stream':
        net = Mobilenetv2_two_stream().to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Resnet18':
        net = Resnet18(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Resnet18_two_stream':
        net = Resnet18_two_stream().to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'LeNet5_one_stream':
        net = LeNet5_one_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'LeNet5_two_stream':
        net = LeNet5_two_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
        
        
        
    # elif Model_name!= 'Flame_one_stream' and Model_name!= 'Flame_two_stream' and MODE == 'both':
    #     print("only Flame networks provide <both> mode")
    #     return
    
    
    print(net)
    
    
    # path_rgb = './254p RGB Images/'
    # path_ir = './254p Thermal Images/'
    path_rgb = args.path_rgb
    path_ir = args.path_ir
    
    Dataset = MyDataset(path_rgb, path_ir,input_size=target_size,transform=Transform_flag)
    
    
    
    Dataset,_ = torch.utils.data.random_split(Dataset, [int(len(Dataset)*args.subset_rate), len(Dataset)-int(len(Dataset)*args.subset_rate)])

    split_rate = args.trainset_rate
    train_set, val_set = torch.utils.data.random_split(Dataset, [int(len(Dataset)*split_rate), len(Dataset)-int(len(Dataset)*split_rate)])
    
    train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)
    
    
    
    optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0.00)
    
    loss_function =nn.CrossEntropyLoss(label_smoothing=0.2)
    correct_on_train = [0]
    correct_on_test = [0]
    
    total_step = (int(len(Dataset)*split_rate)// BATCH_SIZE  )
            
            
    net.train()
    max_accuracy = 0
    Loss_accuracy = []
    test_acc = []
    loss_list = []
    train_acc = []
    
    for index in range(EPOCH):
        net.train()
        for i, (rgb, ir, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            #print(x.shape)
            
            # y_pre = net(rgb.to(DEVICE),ir.to(DEVICE),mode = MODE)
            #print(y_pre.shape)
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE),ir.to(DEVICE),mode = MODE)
            else:
                if MODE=='rgb':
                    x = rgb.to(DEVICE)
                elif  MODE=='ir':
                    x = ir.to(DEVICE)
                    
                y_pre = net(x)
            
            
            
            
            
            
            loss = loss_function(y_pre, y.to(DEVICE))
            Loss_accuracy.append(loss)
            print(f'Epoch:{index + 1}/{EPOCH}     Step:{i+1}|{total_step}   loss:{loss.item()}  ')
            loss_list.append(loss.item())
    
            loss.backward()
    
            optimizer.step()
        
        # if index ==9:
        #     torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
        
        if ((index + 1) % test_interval) == 0:
            # current_accuracy = test(test_dataloader,net,DEVICE,MODE, flag='test_set',output_flag = False)
            current_accuracy =test(test_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = False)
            test_acc.append(current_accuracy)
            current_accuracy_train = test(train_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='train_set',output_flag = False)
            train_acc.append(current_accuracy_train)
            print(f'current max accuracy\t test set:{max(correct_on_test)}%\t train set:{max(correct_on_train)}%')
    
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
    #             torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')
    
    test(test_dataloader,net,DEVICE,MODE,Model_name,name_index,log_path,correct_on_test,correct_on_train,Model_custom_list, flag='test_set',output_flag = True)
    
        # os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
        #           f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')
        
        
        # torch.save(net, f'saved_model/{file_name} final batch={BATCH_SIZE}.pkl')
        
        
    results_array = np.zeros(3,dtype=object)
    results_array[0]= loss_list
    results_array[1]= train_acc
    results_array[2]= test_acc
    
    args.log_loss_path+file_name+'.npy'
    
    np.save(args.log_loss_path+file_name+'.npy',results_array)
    sys.exit()
        
        
    # main()
        
    
# target = torch.tensor([0, 1, 2, 0, 1, 2])
# preds = torch.tensor([0, 2, 1, 0, 0, 1])
# f1_score(preds, target, average ='micro')




    
    
    