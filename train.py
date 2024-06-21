#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
from utils import mean_sq_error
from augmentations import embed_data_mask
import matplotlib.pyplot as plt
from models import SAINT
import torch.optim as optim
import optuna

def train(model, optimizer, scheduler, epochs, trainloader, valloader, trial=None, plot=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_dset = False # porque no es imagen
    criterion = nn.MSELoss().to(device)
    model.to(device)
    valid_rmse = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            loss = criterion(y_outs,y_gts) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            model.eval()
            with torch.no_grad():
                v_rmse = mean_sq_error(model, valloader, device,vision_dset)    
                if trial is not None:
                    trial.report(v_rmse, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            valid_rmse.append(v_rmse)
            model.train()
    mean_valid_rmse = sum(valid_rmse) / len(valid_rmse)


    if plot==True:
        epochs = range(1, len(valid_rmse)+1) 
        plt.figure() 
        plt.title('Learning Curves') 
        plt.xlabel('Epoch') 
        plt.ylabel('RMSE')

        plt.plot(epochs, valid_rmse, 'o-', color='orange', label='Validation RMSE') 
        # plt.plot(epochs, test_rmse, 'o-', color='green', label='Test RMSE') 

        plt.legend(loc='best') 
        plt.grid() 
        plt.show()

    return mean_valid_rmse
    

def build_hidden_mults(base_number):
    return (base_number, base_number // 2)
    
    
def objective(trial, cat_dims, con_idxs, trainloader, validloader, lr=0, wd=0, epochs=0, first_trial=False):  
    if first_trial == True:
        # hyperparameters to be tuned
        dim = 32
        depth = 1
        heads = 4
        attn_dropout = 0.8
        ff_dropout = 0.8
        mlp_hidden_mults = (4, 2)
        # attentiontype = 'colrow'
        final_mlp_style = 'sep'
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1)
        epochs = trial.suggest_int('epochs', 5, 20)
        optimizer = 'AdamW'
        scheduler = 'cosine'

    elif first_trial == False:
        # hyperparameters to be tuned
        dim = trial.suggest_int('dim', 16, 64)
        depth = trial.suggest_int('depth', 1, 3)
        heads = trial.suggest_int('heads', 2, 8)
        attn_dropout = trial.suggest_float('attn_dropout', 0.1, 0.9)
        ff_dropout = trial.suggest_float('ff_dropout', 0.1, 0.9)
        mlp_hidden_mults = trial.suggest_categorical('mlp_hidden_mults', [4, 8, 16])
        mlp_hidden_mults = build_hidden_mults(mlp_hidden_mults)
        # attentiontype = trial.suggest_categorical('attentiontype', ['col','colrow','row','justmlp','attn','attnmlp'])
        final_mlp_style = trial.suggest_categorical('final_mlp_style', ['common','sep'])
        lr = lr
        weight_decay = wd
        epochs = epochs
        optimizer = trial.suggest_categorical('optimizer', ['AdamW','SGD'])
        scheduler = trial.suggest_categorical('scheduler', ['cosine','linear'])
    
    print(f'lr: {lr}, weight_decay: {weight_decay}, epochs: {epochs}, dim: {dim}, depth: {depth}, heads: {heads}, attn_dropout: {attn_dropout}, ff_dropout: {ff_dropout}, mlp_hidden_mults: {mlp_hidden_mults}, final_mlp_style: {final_mlp_style}, optimizer: {optimizer}, scheduler: {scheduler}')

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=dim,
        dim_out=1,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        mlp_hidden_mults=mlp_hidden_mults,
        cont_embeddings='MLP',
        attentiontype='colrow',
        final_mlp_style=final_mlp_style,
        y_dim=1
    )

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not recognized')

    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    elif scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2.667, epochs // 1.6, epochs // 1.142], gamma=0.1)
    
    valid_rmse = train(model, optimizer, scheduler, epochs, trainloader, validloader, trial)

    return valid_rmse
