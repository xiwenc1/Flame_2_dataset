# Flame_2_dataset
The dataset provides rich RGB-thermal image pairs for wildfire flame.

## github address: https://github.com/XiwenChen-Clemson/Flame_2_dataset

![iamge](https://github.com/XiwenChen-Clemson/Flame_2_dataset/blob/main/ex1.PNG)



The codes are for the series CNN baselines tested in our wildfile flame detection dataset.
```

@ARTICLE{9953997, 
author={Chen, Xiwen and Hopkins, Bryce and Wang, Hao and O’Neill, Leo and Afghah, Fatemeh and Razi, Abolfazl and Fulé, Peter and Coen, Janice and Rowell, Eric and Watts, Adam}, 
journal={IEEE Access},  
title={Wildland Fire Detection and Monitoring using a Drone-collected RGB/IR Image Dataset},  
year={2022},  volume={},  number={},  pages={1-1},  
doi={10.1109/ACCESS.2022.3222805}}

@data{FLAME2Dataset,
    doi = {10.21227/swyw-6j78},
    url = {https://dx.doi.org/10.21227/swyw-6j78},
    author = {Hopkins, Bryce and O'Neill, Leo and Afghah, Fatemeh and Razi, Abolfazl and Watts, Adam and Fule, Peter and Coen, Janice},
    publisher = {IEEE Dataport},
    title = {{FLAME} 2: Fire detection and mode{L}ing: {A}erial {M}ulti-spectral imag{E} dataset},
    year = {2022} 
} 
```

## Additional codes for image-process-based fire localization please view: https://github.com/bot0231019/Wildfire-Flame

## How to Run
- Download dataset (254p Frame Pairs.zip (8.30 GB)) from https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset
- Place the data folders (```254p Thermal Images``` and ```254p RGB Images```) at the same work path with your code.
- run ``` rename.py```. This makes the paired images have the same name.
- Then you can run the ```train.py``` to train the model. You can run it by command. 
- ## Some examples are shown in ```eval.ipynb```
- You can also customize your model in ```models.py``` and then import it to ```train.py```
- Some args are shown below, 
```
  ...
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
  ...

```
## We want to mention that the choise of models can be
Please view ```models.py``` to view the detail
``` 
Logistic_two_stream
Flame_one_stream
VGG16
Vgg_two_stream
Logistic
Flame_two_stream
Mobilenetv2
Mobilenetv2_two_stream
LeNet5_one_stream
LeNet5_two_stream
Resnet18
Resnet18_two_stream 
```
The mode can be ``` rbg/ir/both```, however only models listed below support ```both``` mode
```
'Flame_one_stream','Flame_two_stream','Mobilenetv2_two_stream','Vgg_two_stream','Logistic_two_stream','Resnet18_two_stream','LeNet5_one_stream','LeNet5_two_stream'
```
name with one_stream is early-fusion mentioned in the paper while name with two_stream is the late fusion, if applicable. 

Some performance is shown in the below Table,
![iamge](https://github.com/XiwenChen-Clemson/Flame_2_dataset/blob/main/per.PNG)



