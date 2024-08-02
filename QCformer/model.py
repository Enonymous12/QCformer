from functools import partial
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os.path as osp
import os
import numpy as np
import random
import math,time
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

from pathlib import Path
from typing import List, Tuple, Sequence, Optional

from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
import os.path as osp
import math
import torch.nn.functional as F
from utils import RBFExpansion_node,RBFExpansion_edge,RBFExpansion_triangle






device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    






        
class QCBlock(MessagePassing):
    def __init__(self,in_size,out_size):
        super(QCBlock,self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.K_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_v2v)
        self.K_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_e2v)
        self.V_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_v2v)
        self.V_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_e2v)
        
        self.linear_update = torch.nn.Linear(out_size*2,out_size*2,device=device)
        self.layernorm = torch.nn.LayerNorm(out_size*2,device=device)
        self.sigmoid = torch.nn.Sigmoid()
        self.msg_layer = torch.nn.Sequential(torch.nn.Linear(out_size*2,out_size,device=device),torch.nn.LayerNorm(out_size,device=device) )
        
        
        
    def forward(self,x,edge_index,edge_feature):
        K_v = torch.mm(x,self.K_v2v)
        V_v = torch.mm(x,self.V_v2v)
        
        
        if min(edge_feature.shape)==0:
            return V_v
        else:
            out = self.propagate(edge_index,query_v=K_v,key_v=K_v,value_v=V_v,edge_feature=edge_feature)
            return out
        
    
    def message(self,query_v_i,key_v_i,key_v_j,edge_feature,value_v_i,value_v_j):
        K_E = torch.mm(edge_feature,self.K_e2v)
        V_E = torch.mm(edge_feature,self.V_e2v)
        
        query_i = torch.cat([ query_v_i,query_v_i  ],dim=1)
        key_j = torch.cat([ key_v_j,K_E ],dim=1)
        alpha = ( query_i * key_j ) / math.sqrt(self.out_size * 2)
        alpha = F.dropout(alpha,p=0,training=self.training)
        out = torch.cat([ value_v_j,V_E  ],dim=1)
        out = self.linear_update(out) * self.sigmoid( self.layernorm(alpha.view(-1,2*self.out_size)) )
        out = torch.nn.functional.leaky_relu( self.msg_layer(out) )
        return out
        
        



class QCConv(torch.nn.Module):
    def __init__(self,in_size,out_size,head):
        super(QCConv,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.head = head
        self.attention = torch.nn.ModuleList()
        for i in range(self.head):
            self.attention.append(QCBlock(in_size,out_size))
        self.linear_concate_v = torch.nn.Linear(out_size*head,out_size,device=device)
        self.linear_concate_e = torch.nn.Linear(out_size*head,out_size,device=device)
        self.bn_v = torch.nn.BatchNorm1d(out_size,device=device)
        self.bn_e = torch.nn.BatchNorm1d(out_size,device=device)
        self.bn = torch.nn.BatchNorm1d(out_size,device=device)
        
        self.coe1 = torch.nn.Parameter(torch.tensor([0.5],device=device))
        self.coe2 = torch.nn.Parameter(torch.tensor([0.5],device=device))
        
        self.final = torch.nn.Linear(in_size,out_size,device=device)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.linear_concate_v.reset_parameters()
        self.linear_concate_e.reset_parameters()
        self.final.reset_parameters()
        
    
    def forward(self,x,edge_index,edge_feature):
        
        hidden_v = []
        for atten in self.attention:
            hv = atten(x,edge_index,edge_feature)
            hidden_v.append(hv)
            
        hv = torch.cat(hidden_v,dim=1)
        out = self.linear_concate_v(hv)
        out = torch.nn.functional.leaky_relu( self.bn_v(out))
        out = out + x
        return out
    







class QCformer(torch.nn.Module): 
    def __init__(self,in_size,out_size,head1,head2,layer_number):
        super(QCformer,self).__init__()
        self.layer_number = layer_number
        self.in_size = in_size
        self.out_size = out_size
        self.atom_feature_size = 92 
        self.edge_feature_size = 64*3+92*2
        self.triangle_feature_size = 80*3+92*3
        self.atom_init = torch.nn.Sequential(
            RBFExpansion_node() ,
            torch.nn.Linear(self.atom_feature_size,in_size,device=device)
        )
        self.edge_init = torch.nn.Sequential(
            RBFExpansion_edge(vmin=0,vmax=8.0,bins=64),
            torch.nn.Linear(self.edge_feature_size,in_size,device=device)
        )  
        self.triangle_init = torch.nn.Sequential(
            RBFExpansion_triangle(vmin=0,vmax=8.0,bins=80),
            torch.nn.Linear(self.triangle_feature_size,in_size,device=device)
        )  
        
        self.layer1 = torch.nn.ModuleList( [ QCConv(in_size, out_size, head1) for i in range(self.layer_number) ] )
        self.layer2 = torch.nn.ModuleList( [ QCConv(in_size, out_size, head2) for i in range(self.layer_number) ] )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out_size*2, out_size,device=device), torch.nn.LeakyReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(out_size, out_size,device=device),torch.nn.LeakyReLU()
        )
        self.fc_out = torch.nn.Linear(out_size, 1,device=device)
    
        
    def forward(self,data):
        x = self.atom_init(data.x)
        edge_feature = self.edge_init(data.edge_dis)
        triangle_feature = self.triangle_init(data.triangle_dis)
        
        
        for i in range(self.layer_number):
            edge_feature = self.layer2[i](edge_feature,data.triangle_index,triangle_feature)
        
        for i in range(self.layer_number):
            x = self.layer1[i](x,data.edge_index,edge_feature)
        
        
        feature1 = global_mean_pool(x,data.x_batch)
        feature2 = global_mean_pool(edge_feature,data.edge_dis_batch)
        feature = torch.cat([feature1,feature2],dim=1)
        
        
        feature = self.fc(feature)
        feature = self.fc2(feature)
        
        out = self.fc_out(feature)
        return torch.squeeze(out)