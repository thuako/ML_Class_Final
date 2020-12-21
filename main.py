import torch
import torch.nn as nn
import math
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import datetime
import pickle
import os
import matplotlib.pyplot as plt


import utils
import models

if __name__ == "__main__":
    hyper_param_dict = { 
                'root dir': './Result/',
                'project' : 'VGG13',
                'data root' : '../Datasets/cifar10',
                'epochs' : 1,
                'batch' : 256,
                'lr' : 0.05,
                'lr scheduler': 'multi step', # 'multi step', 'step lr', 'cos warm up'
                'step size': 10, # for step lr
                'milestones': [30, 60, 90], # for multi step
                'cycle' : 30, # for cos warm up
                'base min lr' : 0.0001,
                'gamma' : 0.6,
                'optimizer': 'SGD',
                'momentum' : 0.9,   
                'beta1' : 0.9,
                'beta2' : 0.999, 
                'weight_decay' : 5e-4
                }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    



    # run VGG13
    model = models.VGG13()
    project_name = 'VGG13_origin'
    model.to(device)
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run VGG13 ver 1
    model = models.VGG13_v1(conv_drop=0.1, fc_drop=0.2)
    project_name = 'VGG13_v1'
    model.to(device)
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run VGG13 ver 2
    model = models.VGG13_v1(conv_drop=0.3, fc_drop=0.2)
    project_name = 'VGG13_v2'
    model.to(device)
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run GoogleNet without BN
    model = models.GoogLeNet()
    model.to(device)
    project_name = 'GoogLeNet_origin'
    hyper_param_dict['project'] = project_name
    hyper_param_dict['lr'] = 0.05
    hyper_param_dict['batch'] = 200
    # hyper_param_dict['epochs'] = 100
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run GoogleNet without BN
    model = models.GoogLeNet_w_bn()
    model.to(device)
    project_name = 'GoogLeNet_with_BN'
    hyper_param_dict['project'] = project_name
    hyper_param_dict['lr'] = 0.05
    hyper_param_dict['batch'] = 200
    # hyper_param_dict['epochs'] = 100
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    #run GoogleNet ver1
    model = models.GoogLeNet_v1()
    model.to(device)
    project_name = 'GoogLeNet_v1'
    hyper_param_dict['lr scheduler'] = 'cos warm up'
    hyper_param_dict['project'] = project_name
    hyper_param_dict['batch'] = 200
    # hyper_param_dict['epochs'] = 120
    hyper_param_dict['lr'] = 0.05
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run GoogleNet ver 2
    model = models.GoogLeNet_v2()
    model.to(device)
    project_name = 'GoogLeNet_v2'
    hyper_param_dict['lr scheduler'] = 'cos warm up'
    hyper_param_dict['project'] = project_name
    hyper_param_dict['batch'] = 200
    # hyper_param_dict['epochs'] = 120
    hyper_param_dict['lr'] = 0.05
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model


    #run GoogleNet ver3
    model = models.GoogLeNet_v3()
    model.to(device)
    project_name = 'GoogLeNet_v3'
    hyper_param_dict['lr scheduler'] = 'cos warm up'
    hyper_param_dict['project'] = project_name
    hyper_param_dict['batch'] = 200
    # hyper_param_dict['epochs'] = 120
    hyper_param_dict['lr'] = 0.05
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model


    # run ResNet18
    model = models.ResNet18()
    model.to(device)
    project_name = 'ResNet18_origin'
    hyper_param_dict['project'] = project_name  
    hyper_param_dict['batch'] = 256
    hyper_param_dict['lr'] = 0.03
    hyper_param_dict['batch'] = 256
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet34
    model = models.ResNet34()
    model.to(device)
    project_name = 'ResNet34_origin'
    hyper_param_dict['project'] = project_name
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet34
    model = models.ResNet34_v1()
    model.to(device)
    project_name = 'ResNet34_v1'
    hyper_param_dict['project'] = project_name
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet34
    model = models.ResNet34_v2()
    model.to(device)
    project_name = 'ResNet34_v2'
    hyper_param_dict['project'] = project_name
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet34
    model = models.ResNet34_v3()
    model.to(device)
    project_name = 'ResNet34_v3'
    hyper_param_dict['project'] = project_name
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet34
    model = models.ResNet34_v4()
    model.to(device)
    project_name = 'ResNet34_v4'
    hyper_param_dict['project'] = project_name
    print(f'\n\n**************  start new model : {project_name} ******************')
    utils.train(hyper_param_dict, model, device)
    del model

    # run ResNet50
    # model = models.ResNet50()
    # model.to(device)
    # project_name = 'ResNet50'
    # hyper_param_dict['project'] = project_name
    # print(f'\n\n**************  start new model : {project_name} ******************')
    # utils.train(hyper_param_dict, model, device)
    # del model






def load_results():
    #find save dir path
    if hyper_param_dict[ 'project'] is not None:
        save_dir = './Result/' + hyper_param_dict[ 'project'] + '/'
    else:
        save_dir = './Result/'
    dir_list = os.listdir(save_dir)

    for dir_name in dir_list:
        if '_' in dir_name:
            load_dir = save_dir + dir_name
            train_acc = np.load(load_dir + '/np_train_acc_list.npy', allow_pickle=True)
            test_acc = np.load(load_dir + '/np_test_acc_list.npy', allow_pickle=True)
            train_loss = np.load(load_dir + '/np_train_loss_list.npy', allow_pickle=True)
            test_loss = np.load(load_dir + '/np_test_loss_list.npy', allow_pickle=True)

            with open( load_dir + '/hyper.pickle', 'rb') as fr:
                hyper_param_load = pickle.load(fr)

            plt.plot(np.arange(0, train_acc.shape[0]), train_acc, label=  'lr = ' + str(hyper_param_load['lr']))
        plt.legend()
        plt.title(hyper_param_dict['project'])
        #plt.show()