#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch import nn
from utils import mean_sq_error
from augmentations import embed_data_mask
import matplotlib.pyplot as plt
from models import SAINT
import torch.optim as optim

def train(model, optimizer, scheduler, epochs, trainloader, valloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_dset = False # porque no es imagen
    criterion = nn.MSELoss().to(device)
    model.to(device)
    valid_rmse = []
    test_rmse = []
    
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
                ts_rmse = mean_sq_error(model, testloader, device,vision_dset)  
            valid_rmse.append(v_rmse)
            test_rmse.append(ts_rmse)
            model.train()
    
    return valid_rmse, test_rmse

def plot_learning_curve(valid_rmse, test_rmse): 
    epochs = range(1, len(valid_rmse)+1) 

    plt.figure() 
    plt.title('Learning Curves') 
    plt.xlabel('Epoch') 
    plt.ylabel('RMSE')

    plt.plot(epochs, valid_rmse, 'o-', color='orange', label='Validation RMSE') 
    plt.plot(epochs, test_rmse, 'o-', color='green', label='Test RMSE') 

    plt.legend(loc='best') 
    plt.grid() 
    plt.show()
    
    
def objective(trial, cat_dims, con_idxs, trainloader, validloader, testloader):
    
    # hyperparameters to be tuned
    dim = trial.suggest_int('dim', 16, 64)
    depth = trial.suggest_int('depth', 1, 3)
    heads = trial.suggest_int('heads', 2, 8)
    attn_dropout = trial.suggest_float('attn_dropout', 0.1, 0.9)
    ff_dropout = trial.suggest_float('ff_dropout', 0.1, 0.9)
    mlp_hidden_mults = trial.suggest_categorical('mlp_hidden_mults', [(4, 2), (8, 4), (16, 8)])
    attentiontype = trial.suggest_categorical('attentiontype', ['col','colrow','row','justmlp','attn','attnmlp'])
    final_mlp_style = trial.suggest_categorical('final_mlp_style', ['common','sep'])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

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
        attentiontype=attentiontype,
        final_mlp_style=final_mlp_style,
        y_dim=1
    )

    epochs = 100
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = 'cosine'
    
    valid_rmse, test_rmse = train(model, optimizer, scheduler, epochs, trainloader, validloader, testloader)

    return valid_rmse, test_rmse

