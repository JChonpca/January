#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:47:53 2021

@author: iluvatar-JChonpca_Huang
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import random
import os
import time
import shutil
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss, confusion_matrix
from sklearn import metrics

from sko.GA import GA
from sko.PSO import PSO

import warnings
warnings.filterwarnings("ignore")


seed = 577

np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

plt.style.use('seaborn-poster')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


def reproduct_seed(seed=577):
    
    global np
    global random
    global torch
    

    seed = seed
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def service_cpu_run(cpu_num):
    
    cpu_num = cpu_num
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    
    torch.set_num_threads(cpu_num)


class core(nn.Module):
    
    def __init__(self,
                 num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 dropout_rate_each_layers):
        
        super(core, self).__init__()
        
        reproduct_seed()        
        
        self.num_of_input_layer = num_of_input_layer
        
        self.num_of_output_layer = num_of_output_layer
        
        self.num_of_hidden_layer = num_of_hidden_layer
        
        self.node_num_hidden_layers_set = node_num_hidden_layers_set
        
        self.node_num_layers = [self.num_of_input_layer] + self.node_num_hidden_layers_set + [self.num_of_output_layer]
        
        self.act_fun_hidden_layers_set = act_fun_hidden_layers_set
        
        self.dropout_rate_each_layers = dropout_rate_each_layers
        
        self.act_fun = [F.sigmoid, F.relu, F.tanh]
        
        self.layer_list = []
        
        for i in range(self.num_of_hidden_layer):
                        
            self.layer_list.append(nn.Linear(self.node_num_layers[i],self.node_num_layers[i+1]))

        self.layer_list.append(nn.Linear(self.node_num_layers[-2],self.node_num_layers[-1]))
        
        self.layer_list = nn.ModuleList(self.layer_list)
        

    def forward(self, x):
        
        output = x
        
        for i in range(self.num_of_hidden_layer):
            
            output = F.dropout(self.act_fun[self.act_fun_hidden_layers_set[i]](self.layer_list[i](output)),p=self.dropout_rate_each_layers[i])
        
        output = F.dropout(self.layer_list[-1](output),p=self.dropout_rate_each_layers[-1])
        
        return output
    

def read_data(file_x, number_of_input, file_y, number_of_output):
    
    x = np.array(pd.read_excel(file_x,header=None))
    
    y = np.array(pd.read_excel(file_y,header=None))
    
    
    x = x.reshape(-1, number_of_input)
    
    y = y.reshape(-1, number_of_output)
    
    return [x,y]


def reg_data_split(xx,yy):
        
    
    train_x,test_proof_x, train_y, test_proof_y = train_test_split(xx,
                                                       yy,
                                                       test_size = 0.3,
                                                       random_state = seed)
    
    
    
    test_x,proof_x, test_y, proof_y = train_test_split(test_proof_x,
                                                        test_proof_y,
                                                        test_size = 1/3,
                                                        random_state = seed)
    
    
    
    train_x = torch.Tensor(train_x)
    test_x = torch.Tensor(test_x)
    proof_x = torch.Tensor(proof_x)
    
    train_y = torch.Tensor(train_y)
    test_y = torch.Tensor(test_y)
    proof_y = torch.Tensor(proof_y)
    
    
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    proof_x = proof_x.to(device)
    proof_y = proof_y.to(device)
    
    xx = [train_x, test_x, proof_x]
    
    yy = [train_y, test_y, proof_y]
    
    return [xx,yy]


def reg_data_expand(xx,yy,file_x, number_of_input, file_y, number_of_output):
        
    new_data = read_data(file_x, number_of_input, file_y, number_of_output)
    
    tmp_x = new_data[0]
    
    tmp_y = new_data[1]
    
    tmp_x = torch.Tensor(tmp_x)
    
    tmp_y = torch.Tensor(tmp_y)
    
    tmp_x = tmp_x.to(device)
    
    tmp_y = tmp_y.to(device)

    xx[0] = torch.vstack([xx[0],tmp_x])
    
    yy[0] = torch.vstack([yy[0],tmp_y])
    
    return [xx,yy]
    
    
def reg_init_train(net, xx, yy ,total_epoch, batch_size, learning_method, learning_rate, now_time = 'now_time', email = 'localservice'):
    
    
    path = str(int(now_time)) + '_' + email
    
    os.mkdir(path)
    
    os.chdir(path)
    
    os.mkdir('init_train')
    
    os.chdir('init_train')
    
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    EPOCH = total_epoch
    
    BATCH_SIZE = batch_size
    
    LR = learning_rate
        
    train_dataset = Data.TensorDataset(train_x,train_y)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )

    test_dataset = Data.TensorDataset(test_x,test_y)
    
    test_loader = Data.DataLoader(dataset=test_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )


    proof_dataset = Data.TensorDataset(proof_x,proof_y)
    
    proof_loader = Data.DataLoader(dataset=proof_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )
        
    R2_train = []

    R2_test = []
    
    R2_proof = []
    
    loss_train = []
    
    loss_test = []  
    
    loss_proof = []
    
    
    
    optimizers = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.LBFGS]
    
    optimizer =  optimizers[learning_method](net.parameters(), lr=LR)
    
    loss_func = nn.MSELoss()
    
    for epoch in range(EPOCH):
        
        for step, (x, y) in enumerate(train_loader):
            
            net.train()
            
            b_x = x.to(device)
            
            b_y = y.to(device)
            
            def closure():
                
                optimizer.zero_grad()
                
                output = net(b_x)
                
                loss = loss_func(output, b_y)
                
                loss.backward()
            
                return loss
    
            
            optimizer.step(closure)

        net.eval()
            
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

        torch.save(state, str(epoch) + '.pth')
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                train_m = x.cpu().detach().numpy()
                
                output_train = net(x).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
            
            else:
                
                train_m = np.vstack([train_m, x.cpu().detach().numpy()])
                
                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])
                
                
        for step, (x, y) in enumerate(test_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                test_m = x.cpu().detach().numpy()

                output_test = net(x).cpu().detach().numpy()
                
                real_test = y.cpu().detach().numpy()
            
            else:
                
                test_m = np.vstack([test_m, x.detach().cpu().numpy()])

                output_test = np.vstack([output_test,net(x).cpu().detach().numpy()])
                
                real_test = np.vstack([real_test, y.cpu().detach().numpy()])
                

        for step, (x, y) in enumerate(proof_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:

                proof_m = x.cpu().detach().numpy()
                
                output_proof = net(x).cpu().detach().numpy()
                
                real_proof = y.cpu().detach().numpy()
            
            else:

                proof_m = np.vstack([proof_m, x.cpu().detach().numpy()])
                
                output_proof = np.vstack([output_proof, net(x).cpu().detach().numpy()])
                
                real_proof = np.vstack([real_proof, y.cpu().detach().numpy()])
        
        tmp_train_loss = [epoch, mean_squared_error(real_train, output_train)]
        
        tmp_train_r2 = [epoch, r2_score(real_train, output_train)]
        
        tmp_test_loss = [epoch, mean_squared_error(real_test, output_test)]
        
        tmp_test_r2 = [epoch, r2_score(real_test, output_test)]
        
        tmp_proof_loss = [epoch, mean_squared_error(real_proof, output_proof)]
        
        tmp_proof_r2 = [epoch, r2_score(real_proof, output_proof)]
        
                    
        for out_num in range(output_train.shape[1]):
            
            tmp_train_loss.append(mean_squared_error(real_train[:,out_num], output_train[:,out_num]))
            
            tmp_train_r2.append(r2_score(real_train[:,out_num], output_train[:,out_num]))
            
            tmp_test_loss.append(mean_squared_error(real_test[:,out_num], output_test[:,out_num]))
            
            tmp_test_r2.append(r2_score(real_test[:,out_num], output_test[:,out_num]))
            
            tmp_proof_loss.append(mean_squared_error(real_proof[:,out_num], output_proof[:,out_num]))
            
            tmp_proof_r2.append(r2_score(real_proof[:,out_num], output_proof[:,out_num]))           
        
        
        R2_train.append(tmp_train_r2)
        
        R2_test.append(tmp_test_r2)
        
        R2_proof.append(tmp_proof_r2)
        
        loss_train.append(tmp_train_loss)
        
        loss_test.append(tmp_test_loss)
        
        loss_proof.append(tmp_proof_loss)


    R2_train = np.array(R2_train)
    
    R2_test = np.array(R2_test)
    
    R2_proof = np.array(R2_proof)
    
    loss_train = np.array(loss_train)
    
    loss_test = np.array(loss_test)
    
    loss_proof = np.array(loss_proof)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'R2_train.txt', R2_train)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'R2_test.txt', R2_test)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'R2_proof.txt', R2_proof)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_train.txt', loss_train)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_test.txt', loss_test)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_proof.txt', loss_proof)
    
    
    best_model_index = (loss_test[:,1].tolist()).index(min(loss_test[:,1].tolist()))
    
    shutil.copyfile('./' + str(best_model_index) + '.pth', '../init_train_best_' + str(best_model_index) + '.pth')
    
    best_model_path = '../init_train_best_' + str(best_model_index) + '.pth'

    checkpoint = torch.load('./' + str(best_model_index) + '.pth', map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
    
    
    os.chdir('..')
    
    
    return [net, best_model_path, loss_train, loss_test, loss_proof, R2_train, R2_test, R2_proof]



def reg_inter_train(net, net_file, runs_index, xx, yy ,total_epoch, batch_size, learning_method, learning_rate, now_time = 'now_time', email = 'localservice'):
    
    path = str(int(now_time)) + '_' + email
    
    os.mkdir(path)
    
    os.chdir(path)
    
    os.mkdir(str(runs_index) +  '_runs_train')
    
    os.chdir(str(runs_index) +  '_runs_train')
    
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    EPOCH = total_epoch
    
    BATCH_SIZE = batch_size
    
    LR = learning_rate
    
    train_dataset = Data.TensorDataset(train_x,train_y)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )

    test_dataset = Data.TensorDataset(test_x,test_y)
    
    test_loader = Data.DataLoader(dataset=test_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )


    proof_dataset = Data.TensorDataset(proof_x,proof_y)
    
    proof_loader = Data.DataLoader(dataset=proof_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )
        
    R2_train = []

    R2_test = []
    
    R2_proof = []
    
    loss_train = []
    
    loss_test = []  
    
    loss_proof = []
    

    optimizers = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.LBFGS]
    
    optimizer =  optimizers[learning_method](net.parameters(), lr=LR)
    
    loss_func = nn.MSELoss()

    checkpoint = torch.load(net_file, map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
        
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint['epoch']
    
    for epoch in range(start_epoch + 1, start_epoch + EPOCH + 1):
        
        for step, (x, y) in enumerate(train_loader):
            
            net.train()
            
            b_x = x.to(device)
            
            b_y = y.to(device)
            
            def closure():
                
                optimizer.zero_grad()
                
                output = net(b_x)
                
                loss = loss_func(output, b_y)
                
                loss.backward()
            
                return loss
    
            
            optimizer.step(closure)

        net.eval()
            
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

        torch.save(state, str(epoch) + '.pth')
        

        
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                train_m = x.cpu().detach().numpy()
                
                output_train = net(x).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
            
            else:
                
                train_m = np.vstack([train_m, x.cpu().detach().numpy()])
                
                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])
                
                
        for step, (x, y) in enumerate(test_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                test_m = x.cpu().detach().numpy()

                output_test = net(x).cpu().detach().numpy()
                
                real_test = y.cpu().detach().numpy()
            
            else:
                
                test_m = np.vstack([test_m, x.detach().cpu().numpy()])

                output_test = np.vstack([output_test,net(x).cpu().detach().numpy()])
                
                real_test = np.vstack([real_test, y.cpu().detach().numpy()])
                

        for step, (x, y) in enumerate(proof_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:

                proof_m = x.cpu().detach().numpy()
                
                output_proof = net(x).cpu().detach().numpy()
                
                real_proof = y.cpu().detach().numpy()
            
            else:

                proof_m = np.vstack([proof_m, x.cpu().detach().numpy()])
                
                output_proof = np.vstack([output_proof, net(x).cpu().detach().numpy()])
                
                real_proof = np.vstack([real_proof, y.cpu().detach().numpy()])
        
        tmp_train_loss = [epoch, mean_squared_error(real_train, output_train)]
        
        tmp_train_r2 = [epoch, r2_score(real_train, output_train)]
        
        tmp_test_loss = [epoch, mean_squared_error(real_test, output_test)]
        
        tmp_test_r2 = [epoch, r2_score(real_test, output_test)]
        
        tmp_proof_loss = [epoch, mean_squared_error(real_proof, output_proof)]
        
        tmp_proof_r2 = [epoch, r2_score(real_proof, output_proof)]
        
        
        # if output_train.shape[1] != 1:
            
        for out_num in range(output_train.shape[1]):
            
            tmp_train_loss.append(mean_squared_error(real_train[:,out_num], output_train[:,out_num]))
            
            tmp_train_r2.append(r2_score(real_train[:,out_num], output_train[:,out_num]))
            
            tmp_test_loss.append(mean_squared_error(real_test[:,out_num], output_test[:,out_num]))
            
            tmp_test_r2.append(r2_score(real_test[:,out_num], output_test[:,out_num]))
            
            tmp_proof_loss.append(mean_squared_error(real_proof[:,out_num], output_proof[:,out_num]))
            
            tmp_proof_r2.append(r2_score(real_proof[:,out_num], output_proof[:,out_num]))           
        
        
        R2_train.append(tmp_train_r2)
        
        R2_test.append(tmp_test_r2)
        
        R2_proof.append(tmp_proof_r2)
        
        loss_train.append(tmp_train_loss)
        
        loss_test.append(tmp_test_loss)
        
        loss_proof.append(tmp_proof_loss)


    R2_train = np.array(R2_train)
    
    R2_test = np.array(R2_test)
    
    R2_proof = np.array(R2_proof)
    
    loss_train = np.array(loss_train)
    
    loss_test = np.array(loss_test)
    
    loss_proof = np.array(loss_proof)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'R2_train.txt', R2_train)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'R2_test.txt', R2_test)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'R2_proof.txt', R2_proof)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_train.txt', loss_train)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_test.txt', loss_test)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_proof.txt', loss_proof)
    
    best_model_index = (loss_test[:,1].tolist()).index(min(loss_test[:,1].tolist())) + start_epoch + 1
    
    best_model_path = '../ ' +  str(runs_index) +  '_runs_best_' + str(best_model_index) + '.pth'
    
    shutil.copyfile('./' + str(best_model_index) + '.pth', '../ ' +  str(runs_index) +  '_runs_best_' + str(best_model_index) + '.pth')

    checkpoint = torch.load('./' + str(best_model_index) + '.pth', map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])

    os.chdir('..')
    
    
    
    return [net, best_model_path, loss_train, loss_test, loss_proof, R2_train, R2_test, R2_proof]


def reg_net_load(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 dropout_rate_each_layers, 
                 net_file):
    
    net = core(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 dropout_rate_each_layers)

    net.to(device)

    checkpoint = torch.load(net_file, map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
    
    return net


def reg_ensemble_generator(net,
                          xx,
                          yy, 
                          ensemble_num, 
                          batch_size):

    
    
    ENSEMBLE_NUM = ensemble_num
    
    BATCH_SIZE = batch_size
    
    train_dataset = Data.TensorDataset(xx,yy)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=False,
                              batch_size=BATCH_SIZE,                         
                              )
        
    tmp_var = []

    for i in range(ENSEMBLE_NUM):
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                
                output_train = net(x).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
                        
            else:
                            
                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])

        tmp_var.append(output_train)
        
    tmp_var = np.array(tmp_var)
    
    return tmp_var


def reg_eum_cal(ci,pools_ensembles):
    
    
    reg_eum = []
    
    for i in range(pools_ensembles.shape[1]):
        
        tmp_reg_eum = []
        
        for j in range(pools_ensembles.shape[2]):
            
            tmp_var = pools_ensembles[:,i,j]
            
            lower_lim = np.quantile(tmp_var, 0.5-ci/2, axis=0)

            upper_lim = np.quantile(tmp_var, 0.5+ci/2, axis=0)

            tmp_reg_eum.append(upper_lim - lower_lim)
        
        reg_eum.append(tmp_reg_eum)

    reg_eum = np.array(reg_eum)
    
    reg_eum = reg_eum.mean(axis=1).reshape([-1,1])
    
    
    return reg_eum


def reg_loss_dist_cal(net,
                      xx,
                      yy, 
                      batch_size):

    
    BATCH_SIZE = batch_size

    
    train_dataset = Data.TensorDataset(xx,yy)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=False,
                              batch_size=BATCH_SIZE,                         
                              )
    

    for step, (x, y) in enumerate(train_loader):
        
        x = x.to(device)
        
        y = y.to(device)
        
        if step == 0:
            
            
            output_train = net(x).cpu().detach().numpy()
            
            real_train = y.cpu().detach().numpy()
                    
        else:
                        
            output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
            
            real_train = np.vstack([real_train, y.cpu().detach().numpy()])

    q = output_train - real_train
    
    q = q.mean(axis=1).reshape([-1,1])

    return q


def reg_bias_dist_cal(tmp_var):
    
    
    for i in range(tmp_var.shape[1]):
        
        for j in range(tmp_var.shape[2]):
            
            tmp_var[:,i,j] = tmp_var[:,i,j] - tmp_var[:,i,j].mean()
    
    p = tmp_var.mean(axis=2).reshape([-1,1])
    
    
    return p



def reg_dist_align(num_of_input_layer, 
                   num_of_output_layer,
                   num_of_hidden_layer,
                   node_num_hidden_layers_set,
                   act_fun_hidden_layers_set,
                   net_file,
                   runs_index,
                   xx,
                   yy, 
                   ensemble_num, 
                   batch_size,
                   size_pop,
                   max_iter):
    
    
    os.mkdir(str(runs_index) +  '_runs_align')
    
    os.chdir(str(runs_index) +  '_runs_align')
    
    global reg_dist_align_opt_min
    
    train_x = xx[0]
    
    test_x =  xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y =  yy[1]
    
    proof_y = yy[2]
    
    ENSEMBLE_NUM = ensemble_num
    
    BATCH_SIZE = batch_size
    
    net = reg_net_load(num_of_input_layer, 
                       num_of_output_layer,
                       num_of_hidden_layer,
                       node_num_hidden_layers_set,
                       act_fun_hidden_layers_set,
                       [0]*(num_of_hidden_layer+1), 
                       net_file)
        
    
    q = reg_loss_dist_cal(net,
                          train_x,
                          train_y, 
                          batch_size)
    
    
    def reg_dist_align_opt_min(xx):
        
                
        tmp_net = reg_net_load(num_of_input_layer,
                               num_of_output_layer,
                               num_of_hidden_layer,
                               node_num_hidden_layers_set,
                               act_fun_hidden_layers_set,
                               xx, 
                               net_file)
                
        
        tmp_var = reg_ensemble_generator(tmp_net,
                                         train_x,
                                         train_y, 
                                         ENSEMBLE_NUM, 
                                         BATCH_SIZE)
        
        
        p = reg_bias_dist_cal(tmp_var)
            
        tmp_a = plt.hist(p.reshape(-1,1),density=True,label='Bias Distribution')
            
        tmp_b = plt.hist(q.reshape(-1,1),density=True,label='Loss Distribution')
        
        result = ((np.array(tmp_a[0]) - np.array(tmp_b[0]))**2).sum() + ((np.array(tmp_a[1]) - np.array(tmp_b[1]))**2).sum()
        
        plt.legend()
        
        plt.savefig(str(int(time.time())) + '.jpg')
    
        plt.close()
                
        return result
    
    ga = GA(func=reg_dist_align_opt_min, n_dim=num_of_hidden_layer+1, size_pop = size_pop, max_iter = max_iter, lb=[0]*(num_of_hidden_layer+1), ub=[1]*(num_of_hidden_layer+1), precision=1e-7)
    
    best_x, best_y = ga.run()
    
    os.chdir('..')
        
    return best_x
    
    
    
def reg_pool_query(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 net_file,
                 runs_index,
                 xx,
                 yy, 
                 ensemble_num, 
                 batch_size,
                 size_pop,
                 max_iter,
                 pools_x,
                 ci):
        
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    x_label = torch.vstack([train_x, test_x, proof_x]).cpu().detach().numpy()
    
    y_label = torch.vstack([train_y, test_y, proof_y]).cpu().detach().numpy()

    
    best_dropout_rate = reg_dist_align(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 # dropout_rate_each_layers, 
                 net_file,
                 runs_index,
                 xx,
                 yy, 
                 ensemble_num, 
                 batch_size,
                 size_pop,
                 max_iter)
    
    net_file = net_file.split('/')[-1]
    
    best_dropout_net = reg_net_load(num_of_input_layer, 
                                    num_of_output_layer,
                                    num_of_hidden_layer,
                                    node_num_hidden_layers_set,
                                    act_fun_hidden_layers_set,
                                    best_dropout_rate, 
                                    net_file)
    
    pools_x = torch.Tensor(pools_x)
    
    pools_x = pools_x.to(device)
    
    pools_ensembles = reg_ensemble_generator(best_dropout_net,
                                     pools_x,
                                     pools_x, 
                                     ensemble_num, 
                                     batch_size)
    
    pools_x = pools_x.cpu().detach().numpy()
        
    reg_eum = reg_eum_cal(ci, pools_ensembles)
    
    pools_with_eum = np.hstack([pools_x,reg_eum])
    
    pools_with_eum_ranked = pools_with_eum[np.argsort(pools_with_eum[:,-1])]
    
    pools_with_eum_ranked = pools_with_eum_ranked[::-1]
    
    pools_with_ranked = pools_with_eum_ranked[:,:-1]
    
    pools_label_ranked = []
    
    for i in pools_with_ranked:
        
        tmp_index = np.where(x_label == i)[0].shape[0]
        
        if tmp_index == 0:
            
            pools_label_ranked.append(0)
        
        else:
            
            pools_label_ranked.append(1)
    
    pools_label_ranked = np.array(pools_label_ranked)
    
    pools_label_ranked = pools_label_ranked.reshape([pools_x.shape[0],1])
    
    pools_whth_eum_label_ranked = np.hstack([pools_with_eum_ranked, pools_label_ranked])
    
    
    return pools_whth_eum_label_ranked
    

def cls_data_split(xx,yy):
    
    
    train_x,test_proof_x, train_y, test_proof_y = train_test_split(xx,
                                                       yy,
                                                       test_size = 0.3,
                                                       random_state = seed)
    
    
    
    test_x,proof_x, test_y, proof_y = train_test_split(test_proof_x,
                                                        test_proof_y,
                                                        test_size = 1/3,
                                                        random_state = seed)
    
    
    
    train_x = torch.Tensor(train_x)
    test_x = torch.Tensor(test_x)
    proof_x = torch.Tensor(proof_x)
    
    train_y = torch.Tensor(train_y).type(torch.LongTensor)
    test_y = torch.Tensor(test_y).type(torch.LongTensor)
    proof_y = torch.Tensor(proof_y).type(torch.LongTensor)
    
    
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    proof_x = proof_x.to(device)
    proof_y = proof_y.to(device)
    
    xx = [train_x, test_x, proof_x]
    
    yy = [train_y, test_y, proof_y]
    
    return [xx,yy]


def cls_data_expand(xx,yy,file_x, number_of_input, file_y, number_of_output):
        
    new_data = read_data(file_x, number_of_input, file_y, number_of_output)
    
    tmp_x = new_data[0]
    
    tmp_y = new_data[1]
    
    tmp_x = torch.Tensor(tmp_x)
    
    tmp_y = torch.Tensor(tmp_y).type(torch.LongTensor)
    
    tmp_x = tmp_x.to(device)
    
    tmp_y = tmp_y.to(device)

    xx[0] = torch.vstack([xx[0],tmp_x])
    
    yy[0] = torch.hstack([yy[0],tmp_y])
    
    return [xx,yy]




def cls_init_train(net, xx, yy ,total_epoch, batch_size, learning_method, learning_rate, now_time = 'now_time', email = 'localservice'):
    
    
    path = str(int(now_time)) + '_' + email
    
    os.mkdir(path)
    
    os.chdir(path)
    
    os.mkdir('init_train')
    
    os.chdir('init_train')
    
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    EPOCH = total_epoch
    
    BATCH_SIZE = batch_size
    
    LR = learning_rate
    
    train_dataset = Data.TensorDataset(train_x,train_y)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )

    test_dataset = Data.TensorDataset(test_x,test_y)
    
    test_loader = Data.DataLoader(dataset=test_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )


    proof_dataset = Data.TensorDataset(proof_x,proof_y)
    
    proof_loader = Data.DataLoader(dataset=proof_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )
        
    confusion_train = []

    confusion_test = []
    
    confusion_proof = []
    
    loss_train = []
    
    loss_test = []  
    
    loss_proof = []
    
    optimizers = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.LBFGS]
    
    optimizer =  optimizers[learning_method](net.parameters(), lr=LR)
    
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH):
        
        for step, (x, y) in enumerate(train_loader):
            
            net.train()
            
            b_x = x.to(device)
            
            b_y = y.to(device)
            
            def closure():
                
                optimizer.zero_grad()
                
                output = net(b_x)
                
                loss = loss_func(output, b_y)
                
                loss.backward()
            
                return loss
    
            
            optimizer.step(closure)

        net.eval()
            
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

        torch.save(state, str(epoch) + '.pth')
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                train_m = x.cpu().detach().numpy()
                
                output_train = net(x).cpu().detach().numpy()
                                
                output_train_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
            
            else:
                
                train_m = np.vstack([train_m, x.cpu().detach().numpy()])

                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                output_train_cls = np.vstack([output_train_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])
                
                
        for step, (x, y) in enumerate(test_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                test_m = x.cpu().detach().numpy()
                
                output_test = net(x).cpu().detach().numpy()

                output_test_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_test = y.cpu().detach().numpy()
            
            else:
                
                test_m = np.vstack([test_m, x.detach().cpu().numpy()])

                output_test = np.vstack([output_test, net(x).cpu().detach().numpy()])

                output_test_cls = np.vstack([output_test_cls,(torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_test = np.vstack([real_test, y.cpu().detach().numpy()])
                

        for step, (x, y) in enumerate(proof_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:

                proof_m = x.cpu().detach().numpy()
                
                output_proof = net(x).cpu().detach().numpy()
                                
                output_proof_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_proof = y.cpu().detach().numpy()
            
            else:

                proof_m = np.vstack([proof_m, x.cpu().detach().numpy()])

                output_proof = np.vstack([output_proof, net(x).cpu().detach().numpy()])
                
                output_proof_cls = np.vstack([output_proof_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_proof = np.vstack([real_proof, y.cpu().detach().numpy()])
        
        tmp_train_loss = [epoch, log_loss(real_train, output_train)]
        
        tmp_train_confusion = [epoch] + confusion_matrix(real_train, output_train_cls).reshape(-1).tolist()
        
        tmp_test_loss = [epoch, log_loss(real_test, output_test)]
        
        tmp_test_confusion = [epoch] + confusion_matrix(real_test, output_test_cls).reshape(-1).tolist()
        
        tmp_proof_loss = [epoch, log_loss(real_proof, output_proof)]
        
        tmp_proof_confusion = [epoch] + confusion_matrix(real_proof, output_proof_cls).reshape(-1).tolist()
        
        confusion_train.append(tmp_train_confusion)
        
        confusion_test.append(tmp_test_confusion)
        
        confusion_proof.append(tmp_proof_confusion)
        
        loss_train.append(tmp_train_loss)
        
        loss_test.append(tmp_test_loss)
        
        loss_proof.append(tmp_proof_loss)


    confusion_train = np.array(confusion_train)
    
    confusion_test = np.array(confusion_test)
    
    confusion_proof = np.array(confusion_proof)
    
    loss_train = np.array(loss_train)
    
    loss_test = np.array(loss_test)
    
    loss_proof = np.array(loss_proof)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'confusion_train.txt', confusion_train)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'confusion_test.txt', confusion_test)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'confusion_proof.txt', confusion_proof)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_train.txt', loss_train)

    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_test.txt', loss_test)
    
    np.savetxt('0-' + str(EPOCH-1) + '_' + 'loss_proof.txt', loss_proof)
    
    
    best_model_index = (loss_test[:,1].tolist()).index(min(loss_test[:,1].tolist()))
    
    shutil.copyfile('./' + str(best_model_index) + '.pth', '../init_train_best_' + str(best_model_index) + '.pth')
    
    best_model_path = '../init_train_best_' + str(best_model_index) + '.pth'

    checkpoint = torch.load('./' + str(best_model_index) + '.pth', map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
    
    
    os.chdir('..')
    
    
    return [net, best_model_path, loss_train, loss_test, loss_proof, confusion_train, confusion_test, confusion_proof]



def cls_inter_train(net, net_file, runs_index, xx, yy ,total_epoch, batch_size, learning_method, learning_rate, now_time = 'now_time', email = 'localservice'):
    
    path = str(int(now_time)) + '_' + email
    
    os.mkdir(path)
    
    os.chdir(path)
    
    os.mkdir(str(runs_index) +  '_runs_train')
    
    os.chdir(str(runs_index) +  '_runs_train')
    
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    EPOCH = total_epoch
    
    BATCH_SIZE = batch_size
    
    LR = learning_rate
    
    train_dataset = Data.TensorDataset(train_x,train_y)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )

    test_dataset = Data.TensorDataset(test_x,test_y)
    
    test_loader = Data.DataLoader(dataset=test_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )


    proof_dataset = Data.TensorDataset(proof_x,proof_y)
    
    proof_loader = Data.DataLoader(dataset=proof_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,                         
                              )
        
    confusion_train = []

    confusion_test = []
    
    confusion_proof = []
    
    loss_train = []
    
    loss_test = []  
    
    loss_proof = []
    

    optimizers = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.LBFGS]
    
    optimizer =  optimizers[learning_method](net.parameters(), lr=LR)
    
    loss_func = nn.MSELoss()

    checkpoint = torch.load(net_file, map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
        
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint['epoch']
    
    for epoch in range(start_epoch + 1, start_epoch + EPOCH + 1):
        
        for step, (x, y) in enumerate(train_loader):
            
            net.train()
            
            b_x = x.to(device)
            
            b_y = y.to(device)
            
            def closure():
                
                optimizer.zero_grad()
                
                output = net(b_x)
                
                loss = loss_func(output, b_y)
                
                loss.backward()
            
                return loss
    
            
            optimizer.step(closure)

        net.eval()
            
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

        torch.save(state, str(epoch) + '.pth')
        

        
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                train_m = x.cpu().detach().numpy()
                
                output_train = net(x).cpu().detach().numpy()
                                
                output_train_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
            
            else:
                
                train_m = np.vstack([train_m, x.cpu().detach().numpy()])

                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                output_train_cls = np.vstack([output_train_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])
                
                
        for step, (x, y) in enumerate(test_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                test_m = x.cpu().detach().numpy()
                
                output_test = net(x).cpu().detach().numpy()

                output_test_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_test = y.cpu().detach().numpy()
            
            else:
                
                test_m = np.vstack([test_m, x.detach().cpu().numpy()])

                output_test = np.vstack([output_test, net(x).cpu().detach().numpy()])

                output_test_cls = np.vstack([output_test_cls,(torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_test = np.vstack([real_test, y.cpu().detach().numpy()])
                

        for step, (x, y) in enumerate(proof_loader):

            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:

                proof_m = x.cpu().detach().numpy()
                
                output_proof = net(x).cpu().detach().numpy()
                                
                output_proof_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_proof = y.cpu().detach().numpy()
            
            else:

                proof_m = np.vstack([proof_m, x.cpu().detach().numpy()])

                output_proof = np.vstack([output_proof, net(x).cpu().detach().numpy()])
                
                output_proof_cls = np.vstack([output_proof_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_proof = np.vstack([real_proof, y.cpu().detach().numpy()])
        
        tmp_train_loss = [epoch, log_loss(real_train, output_train)]
        
        tmp_train_confusion = [epoch] + confusion_matrix(real_train, output_train_cls).reshape(-1).tolist()
        
        tmp_test_loss = [epoch, log_loss(real_test, output_test)]
        
        tmp_test_confusion = [epoch] + confusion_matrix(real_test, output_test_cls).reshape(-1).tolist()
        
        tmp_proof_loss = [epoch, log_loss(real_proof, output_proof)]
        
        tmp_proof_confusion = [epoch] + confusion_matrix(real_proof, output_proof_cls).reshape(-1).tolist()
        
        
        confusion_train.append(tmp_train_confusion)
        
        confusion_test.append(tmp_test_confusion)
        
        confusion_proof.append(tmp_proof_confusion)
        
        loss_train.append(tmp_train_loss)
        
        loss_test.append(tmp_test_loss)
        
        loss_proof.append(tmp_proof_loss)


    confusion_train = np.array(confusion_train)
    
    confusion_test = np.array(confusion_test)
    
    confusion_proof = np.array(confusion_proof)
    
    loss_train = np.array(loss_train)
    
    loss_test = np.array(loss_test)
    
    loss_proof = np.array(loss_proof)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'confusion_train.txt', confusion_train)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'confusion_test.txt', confusion_test)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'confusion_proof.txt', confusion_proof)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_train.txt', loss_train)

    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_test.txt', loss_test)
    
    np.savetxt( str(start_epoch + 1) + '-' + str(start_epoch + EPOCH + 1 -1) + '_' + 'loss_proof.txt', loss_proof)
    
    best_model_index = (loss_test[:,1].tolist()).index(min(loss_test[:,1].tolist())) + start_epoch + 1
    
    best_model_path = '../ ' +  str(runs_index) +  '_runs_best_' + str(best_model_index) + '.pth'
    
    shutil.copyfile('./' + str(best_model_index) + '.pth', '../ ' +  str(runs_index) +  '_runs_best_' + str(best_model_index) + '.pth')

    checkpoint = torch.load('./' + str(best_model_index) + '.pth', map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])

    os.chdir('..')
    
    
    
    return [net, best_model_path, loss_train, loss_test, loss_proof, confusion_train, confusion_test, confusion_proof]



def cls_net_load(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 dropout_rate_each_layers, 
                 net_file):
    
    net = core(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 dropout_rate_each_layers)

    net.to(device)

    checkpoint = torch.load(net_file, map_location='cpu')
    
    net.load_state_dict(checkpoint['model'])
    
    return net


def cls_ensemble_generator(net,
                          xx,
                          yy, 
                          ensemble_num, 
                          batch_size):

    
    
    ENSEMBLE_NUM = ensemble_num
    
    BATCH_SIZE = batch_size
    
    train_dataset = Data.TensorDataset(xx,yy)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=False,
                              batch_size=BATCH_SIZE,                         
                              )
        
    tmp_var = []
    
    for i in range(ENSEMBLE_NUM):
        
        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            
            y = y.to(device)
            
            if step == 0:
                
                train_m = x.cpu().detach().numpy()
                
                output_train = net(x).cpu().detach().numpy()
                                
                output_train_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
                
                real_train = y.cpu().detach().numpy()
            
            else:
                
                train_m = np.vstack([train_m, x.cpu().detach().numpy()])

                output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
                
                output_train_cls = np.vstack([output_train_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
                
                real_train = np.vstack([real_train, y.cpu().detach().numpy()])
        
        tmp_var.append(output_train)
        
    tmp_var = np.array(tmp_var)
    
    return tmp_var


def cls_eum_cal(ci,pools_ensembles):
    
    
    cls_eum = []
    
    for i in range(pools_ensembles.shape[1]):
        
        cls_eum.append(pools_ensembles[:,i,:].mean(axis=0).max())
        
    cls_eum = np.array(cls_eum).reshape([-1,1])
        
    return cls_eum


def cls_loss_dist_cal(net,
                      xx,
                      yy, 
                      batch_size):
    
    BATCH_SIZE = batch_size

    
    train_dataset = Data.TensorDataset(xx,yy)
    
    train_loader = Data.DataLoader(dataset=train_dataset,
                              shuffle=False,
                              batch_size=BATCH_SIZE,                         
                              )
    
    for step, (x, y) in enumerate(train_loader):
        
        x = x.to(device)
        
        y = y.to(device)
        
        if step == 0:
            
            train_m = x.cpu().detach().numpy()
            
            output_train = net(x).cpu().detach().numpy()
                            
            output_train_cls = (torch.max(net(x),1)[1]).cpu().detach().numpy()
            
            real_train = y.cpu().detach().numpy()
        
        else:
            
            train_m = np.vstack([train_m, x.cpu().detach().numpy()])

            output_train = np.vstack([output_train, net(x).cpu().detach().numpy()])
            
            output_train_cls = np.vstack([output_train_cls, (torch.max(net(x),1)[1]).cpu().detach().numpy()])
            
            real_train = np.vstack([real_train, y.cpu().detach().numpy()])

    
    right_index = (output_train_cls == real_train)
    
    right_output_train = output_train[right_index]
    
    right_output_train_cls = output_train_cls[right_index]
    
    right_real_train = real_train[right_index]
    
    avg_ci = []
    
    for i in range(output_train.shape[1]):
        
        tmp_right_output_train_cls = (right_output_train_cls == i)
        
        tmp_right_output_train = right_output_train[tmp_right_output_train_cls]
        
        avg_ci.append(tmp_right_output_train[:,i].mean())
    
    q = []
    
    for i in range(output_train.shape[0]):
        
        tmp_cls = output_train_cls[i]
        
        q.append( output_train[i,tmp_cls] - avg_ci[tmp_cls])
    
    
    q = np.array(q).reshape([-1,1])
    
    return q



def cls_bias_dist_cal(tmp_var):
    
    p = []
    
    for i in range(tmp_var.shape[1]):
        
        tmp_cls = torch.max(torch.Tensor([tmp_var[:,i,:].mean(axis=0)]), 1)[1].numpy()[0]
        
        tmp_p = tmp_var[:,i,tmp_cls] - tmp_var[:,i,:].mean(axis=0)[tmp_cls]
        
        p.append(tmp_p.reshape(-1,1))
    
    p = np.array(p).reshape([-1,1])
    
    return p



def cls_dist_align(num_of_input_layer, 
                   num_of_output_layer,
                   num_of_hidden_layer,
                   node_num_hidden_layers_set,
                   act_fun_hidden_layers_set,
                   net_file,
                   runs_index,
                   xx,
                   yy, 
                   ensemble_num, 
                   batch_size,
                   size_pop,
                   max_iter):
    
    
    os.mkdir(str(runs_index) +  '_runs_align')
    
    os.chdir(str(runs_index) +  '_runs_align')
    
    global cls_dist_align_opt_min
    
    train_x = xx[0]
    
    test_x =  xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y =  yy[1]
    
    proof_y = yy[2]
    
    ENSEMBLE_NUM = ensemble_num
    
    BATCH_SIZE = batch_size
    
    net = cls_net_load(num_of_input_layer, 
                       num_of_output_layer,
                       num_of_hidden_layer,
                       node_num_hidden_layers_set,
                       act_fun_hidden_layers_set,
                       [0]*(num_of_hidden_layer+1), 
                       net_file)
        
    
    q = cls_loss_dist_cal(net,
                          train_x,
                          train_y, 
                          batch_size)
    
    
    def cls_dist_align_opt_min(xx):
        
                
        tmp_net = cls_net_load(num_of_input_layer,
                               num_of_output_layer,
                               num_of_hidden_layer,
                               node_num_hidden_layers_set,
                               act_fun_hidden_layers_set,
                               xx, 
                               net_file)
                
        
        tmp_var = cls_ensemble_generator(tmp_net,
                                         train_x,
                                         train_y, 
                                         ENSEMBLE_NUM, 
                                         BATCH_SIZE)
        
        
        p = cls_bias_dist_cal(tmp_var)
            
        tmp_a = plt.hist(p.reshape(-1,1),density=True,label='Bias Distribution')
            
        tmp_b = plt.hist(q.reshape(-1,1),density=True,label='Loss Distribution')
        
        result = ((np.array(tmp_a[0]) - np.array(tmp_b[0]))**2).sum() + ((np.array(tmp_a[1]) - np.array(tmp_b[1]))**2).sum()
        
        plt.legend()
        
        plt.savefig(str(int(time.time())) + '.jpg')
    
        plt.close()
                
        return result
    
    ga = GA(func=cls_dist_align_opt_min, n_dim=num_of_hidden_layer+1, size_pop = size_pop, max_iter = max_iter, lb=[0]*(num_of_hidden_layer+1), ub=[1]*(num_of_hidden_layer+1), precision=1e-7)
    
    best_x, best_y = ga.run()
    
    os.chdir('..')
        
    return best_x
    
    
    
def cls_pool_query(num_of_input_layer, 
                 num_of_output_layer,
                 num_of_hidden_layer,
                 node_num_hidden_layers_set,
                 act_fun_hidden_layers_set,
                 net_file,
                 runs_index,
                 xx,
                 yy, 
                 ensemble_num, 
                 batch_size,
                 size_pop,
                 max_iter,
                 pools_x,
                 ci):
        
    train_x = xx[0]
    
    test_x = xx[1]
    
    proof_x = xx[2]
    
    train_y = yy[0]
    
    test_y = yy[1]
    
    proof_y = yy[2]
    
    x_label = torch.vstack([train_x, test_x, proof_x]).cpu().detach().numpy()
    
    y_label = torch.hstack([train_y, test_y, proof_y]).cpu().detach().numpy()

    
    best_dropout_rate = cls_dist_align(num_of_input_layer, 
                                       num_of_output_layer,
                                       num_of_hidden_layer,
                                       node_num_hidden_layers_set,
                                       act_fun_hidden_layers_set,
                                       # dropout_rate_each_layers, 
                                       net_file,
                                       runs_index,
                                       xx,
                                       yy, 
                                       ensemble_num, 
                                       batch_size,
                                       size_pop,
                                       max_iter)
    
    net_file = net_file.split('/')[-1]
    
    best_dropout_net = cls_net_load(num_of_input_layer, 
                                    num_of_output_layer,
                                    num_of_hidden_layer,
                                    node_num_hidden_layers_set,
                                    act_fun_hidden_layers_set,
                                    best_dropout_rate, 
                                    net_file)
    
    pools_x = torch.Tensor(pools_x)
    
    pools_x = pools_x.to(device)
    
    pools_ensembles = cls_ensemble_generator(best_dropout_net,
                                     pools_x,
                                     pools_x, 
                                     ensemble_num, 
                                     batch_size)
    
    pools_x = pools_x.cpu().detach().numpy()
        
    cls_eum = cls_eum_cal(ci, pools_ensembles)
    
    pools_with_eum = np.hstack([pools_x,cls_eum])
    
    pools_with_eum_ranked = pools_with_eum[np.argsort(pools_with_eum[:,-1])]
    
    # pools_with_eum_ranked = pools_with_eum_ranked[::-1]
    
    pools_with_ranked = pools_with_eum_ranked[:,:-1]
    
    pools_label_ranked = []
    
    for i in pools_with_ranked:
        
        tmp_index = np.where(x_label == i)[0].shape[0]
        
        if tmp_index == 0:
            
            pools_label_ranked.append(0)
        
        else:
            
            pools_label_ranked.append(1)
    
    pools_label_ranked = np.array(pools_label_ranked)
    
    pools_label_ranked = pools_label_ranked.reshape([pools_x.shape[0],1])
    
    pools_whth_eum_label_ranked = np.hstack([pools_with_eum_ranked, pools_label_ranked])
    
    
    return pools_whth_eum_label_ranked
