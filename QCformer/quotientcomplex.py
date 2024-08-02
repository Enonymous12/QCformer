import pandas as pd
from functools import partial
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os.path as osp
import os
import numpy as np
import random
import math,time
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.core.atoms import Atoms
from pathlib import Path
from typing import Optional
from typing import List, Tuple, Sequence, Optional




from pandarallel import pandarallel
pandarallel.initialize()


def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        classification=False,
        id_tag="id",
        neighbor_strategy="",
        lineControl=True,
        mean_train=None,
        std_train=None,
        dataname=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        if dataname=='jarvis':
            self.ids = self.df['jid']
        elif dataname=='mp':
            self.ids = self.df['id']
        self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        #print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        
        
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            #print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            #print("normalize using training mean %f and std %f" % (mean_train, std_train))
        
        #print('start transform')
        self.transform = transform

        
        
        
        '''
        #use new feature to replace this part
        
        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f
        '''
        
        
        #print('start batch')
        self.prepare_batch = prepare_pyg_batch
        #print('batch ok')

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)
        


def area_from_edges(dis):
    dis = [ round(dis[0],4),round(dis[1],4),round(dis[2],4) ]
    if abs(dis[0]+dis[1]-dis[2])<=0.01 or abs(dis[0]+dis[2]-dis[1])<=0.01 or abs(dis[1]+dis[2]-dis[0])<=0.01:
        return 0
    p = ( dis[0] +dis[1] +dis[2] )/2
    return pow( p * (p-dis[0]) * (p-dis[1]) * (p-dis[2]) ,0.5)

def perimeter_from_edges(dis):
    return dis[0]+dis[1]+dis[2]

    
    
def angle_from_edges(dis,i):
    temp = 0
    index = []
    for j in [0,1,2]:
        if j!=i:
            index.append(j)
            temp = temp + pow(dis[j],2)
    temp = temp - pow(dis[i],2)
    temp = round( temp/(2*dis[ index[0] ]*dis[ index[1] ]),2)
    return math.acos(temp)
    
def all_from_edges(dis):
    temp1 = area_from_edges(dis)
    temp2 = perimeter_from_edges(dis)
    ang1 = angle_from_edges(dis,0)
    ang2 = angle_from_edges(dis,1)
    ang3 = angle_from_edges(dis,2)
    return [temp1,temp2,dis[0],dis[1],dis[2],ang1,ang2,ang3]
    

def change_to_onehot2(a,n):
    a = int(a/0.1)
    if a>=n:
        a = int(n-1)
    temp = [0] * n
    temp[a] = 1
    return temp
    
def rbf(dis,vmin,vmax,bins):
    #vmin = 0
    #vmax = 10
    #bins = 150
    center = np.linspace(vmin, vmax, bins)
    lengthscale = np.diff(center).mean()
    gamma = 1 / lengthscale
    res = []
    for v in center:
        temp = np.exp( -gamma*(dis-v)**2 )
        res.append(temp)
    M = max(res)
    for i in range(len(res)):
        res[i] = res[i]/M
    return res

    
        


def get_complex(atoms,cutoff=8,max_neighbors=12):
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return get_complex(atoms=atoms,cutoff=r_cut,max_neighbors=max_neighbors)
    
    
    atom_name = []
    for ii, s in enumerate(atoms.elements):
        atom_name.append(s)
    
    
    
    # point
    new_point = []
    point_dict = {}
    for i in range(atoms.num_atoms):
        new_point.append([i,0,0,0])
        point_dict[str([ i,0,0,0 ])] = i
    pn = atoms.num_atoms
    for item in all_neighbors:
        neighbor = sorted(item,key=lambda x: x[2])
        for i in range(max_neighbors):
            one = neighbor[i][0]
            two = neighbor[i][1]
            val = neighbor[i][2]
            coor = neighbor[i][3]
            temp = [two,round(coor[0]),round(coor[1]),round(coor[2])]
            if str(temp) not in point_dict:
                new_point.append(temp)
                atom_name.append( atom_name[two] )
                point_dict[str(temp)] = pn
                pn = pn + 1
                
    
    
    
    
    # edge
    edge_index = [ [],[] ]
    edge_dis = []
    E = []
    E_dict = {}
    en = 0
    for item in all_neighbors:
        neighborlist = sorted(item, key=lambda x: x[2])       
        for i in range(max_neighbors):
            one = neighborlist[i][0]
            two = neighborlist[i][1]
            val = neighborlist[i][2]
            coor = neighborlist[i][3]
            
            edge_index[0].append(two)
            edge_index[1].append(one)
            temp = point_dict[str([ two,round(coor[0]),round(coor[1]),round(coor[2]) ])]
            E.append([temp,one])
            a1 = get_node_attributes(atom_name[one],atom_features='atomic_number')
            a2 = get_node_attributes(atom_name[two],atom_features='atomic_number')
            edge_dis.append(a1+a2+[val])
            E_dict[str([temp,one])] = [en,val]
            en = en + 1
            
            
                
        
    
    # triangle
    T = []
    T_dis = []
    triangle_index = [ [],[] ]
    triangle_dis = []
    for i in range(atoms.num_atoms,len(new_point)):
        #feat1 = get_cgcnn_value(atom_name[i],30) # 47 + 5N
        #feat1 = list(get_node_attributes(atom_name[i], atom_features='cgcnn'))
        feat1 = get_node_attributes(atom_name[i],atom_features='atomic_number')
        for j in range(atoms.num_atoms):
            if str([i,j]) in E_dict:
                dis1 = E_dict[str([i,j])][1]
                #feat2 = get_cgcnn_value(atom_name[j],30) # 47 + 5N
                #feat2 = list(get_node_attributes(atom_name[j], atom_features='cgcnn'))
                feat2 = get_node_attributes(atom_name[j],atom_features='atomic_number')
                for k in range(atoms.num_atoms):
                    #feat3 = get_cgcnn_value(atom_name[k],30) # 47 + 5N
                    #feat3 = list(get_node_attributes(atom_name[k], atom_features='cgcnn'))
                    feat3 = get_node_attributes(atom_name[k],atom_features='atomic_number')
                    if j!=k and str([i,k]) in E_dict:
                        dis2 = E_dict[str([i,k])][1]
                        if str([j,k]) in E_dict:
                            
                            dis3 = E_dict[str([j,k])][1]
                            T.append([i,j,k])
                            #T_dis.append([dis1,dis2,dis3])
                            feat4 = area_from_edges([dis1,dis2,dis3])
                            T_dis.append(feat1+feat2+feat3+[feat4])
                            
                            
                            
                        if str([k,j]) in E_dict:
                            dis3 = E_dict[str([k,j])][1]
                            T.append([i,k,j])
                            #T_dis.append([dis2,dis1,dis3])
                            feat4 = area_from_edges([dis2,dis1,dis3])
                            T_dis.append(feat1+feat3+feat2+[feat4])
                            
    for i in range(len(T)):
        one = T[i][0]
        two = T[i][1]
        thr = T[i][2]
        
        index1 = E_dict[str([one,two])][0]
        index2 = E_dict[str([one,thr])][0]
        index3 = E_dict[str([two,thr])][0]
        
        triangle_index[0].append(index1)
        triangle_index[1].append(index2)
        triangle_dis.append(T_dis[i])
        #triangle_dis.append( angle_from_edges(T_dis[i],2) )
        
        
        triangle_index[0].append(index1)
        triangle_index[1].append(index3)
        triangle_dis.append(T_dis[i])
        #triangle_dis.append( angle_from_edges(T_dis[i],1) )
        
        triangle_index[0].append(index2)
        triangle_index[1].append(index3)
        triangle_dis.append(T_dis[i])
        #triangle_dis.append( angle_from_edges(T_dis[i],0) )
    #for i in range(len(triangle_dis)):
        #triangle_dis[i] = (triangle_dis[i][0]+triangle_dis[i][1]+triangle_dis[i][2])/3
        #triangle_dis[i] = area_from_edges(triangle_dis[i])
        #triangle_dis[i] = perimeter_from_edges(triangle_dis[i])
        #triangle_dis[i] = all_from_edges(triangle_dis[i])
        
    
    
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    edge_dis = torch.tensor(edge_dis,dtype=torch.get_default_dtype())
    triangle_index = torch.tensor(triangle_index,dtype=torch.long)
    triangle_dis = torch.tensor(triangle_dis,dtype=torch.get_default_dtype())
    return edge_index,edge_dis,triangle_index,triangle_dis
    
    



            
def change_to_N(value,a1,a2,N):
    k = N/(a2-a1)
    b = -a1 * k
    return value * k + b
            
def change_to_onehot(a,n):
    if a<0 or a>n:
        print(a,n)
        print('error')
    if a==n:
        a = n-1
    temp = [0] * n
    temp[int(a)] = 1
    return temp
    
            
def get_cgcnn_value(ele,N):
    group = 0
    if ele in [ 'H','Li','Na','K','Rb','Cs','Fr' ]:
        group = 1
    elif ele in [ 'Be','Mg','Ca','Sr','Ba','Ra' ]:
        group = 2
    elif ele in [ 'Sc','Y','Lu','Lr','La','Ac' ]:
        group = 3
    elif ele in [ 'Ti','Zr','Hf','Rf','Ce','Th' ]:
        group = 4
    elif ele in [ 'V','Nb','Ta','Db','Pr','Pa' ]:
        group = 5
    elif ele in [ 'Cr','Mo','W','Sg','Nd','U']:
        group = 6
    elif ele in [ 'Mn','Tc','Re','Bh','Pm','Np' ]:
        group = 7
    elif ele in [ 'Fe','Ru','Os','Hs','Sm','Pu' ]:
        group = 8
    elif ele in [ 'Co','Rh','Ir','Mt','Eu','Am' ]:
        group = 9
    elif ele in [ 'Ni','Pd','Pt','Ds','Gd','Cm' ]:
        group = 10
    elif ele in [ 'Cu','Ag','Au','Rg','Tb','Bk' ]:
        group = 11
    elif ele in [ 'Zn','Cd','Hg','Cn','Dy','Cf' ]:
        group = 12
    elif ele in [ 'B','Al','Ga','In','Tl','Nh','Ho','Es' ]:
        group = 13
    elif ele in [ 'C','Si','Ge','Sn','Pb','Fl','Er','Fm' ]:
        group = 14
    elif ele in [ 'N','P','As','Sb','Bi','Mc','Tm','Md' ]:
        group = 15
    elif ele in [ 'O','S','Se','Te','Po','Lv','Yb','No' ]:
        group = 16
    elif ele in [ 'F','Cl','Br','I','At','Ts' ]:
        group = 17
    elif ele in [ 'He','Ne','Ar','Kr','Xe','Rn','Og' ]:
        group = 18
    else:
        print('error')
        return
    
    period = 0
    if ele in [ 'H','He' ]:
        period = 1
    elif ele in [ 'Li','Be','B','C','N','O','F','Ne' ]:
        period = 2
    elif ele in [ 'Na','Mg','Al','Si','P','S','Cl','Ar' ]:
        period = 3
    elif ele in [ 'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr' ]:
        period = 4
    elif ele in [ 'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe' ]:
        period = 5
    elif ele in [ 'Cs','Ba','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']:
        period = 6
    elif ele in [ 'Fr','Ra','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og' ]:
        period = 7
    elif ele in [ 'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb' ]:
        period = 8
    elif ele in [ 'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No' ]:
        period = 9
    else:
        print('error')
        return
    
    block = 0
    if group in [1,2] or ele=='He':
        block = 1
    elif group in [3,4,5,6,7,8,9,10,11,12] and period in [4,5,6,7]:
        block = 2
    elif group in [13,14,15,16,17,18] and period in [2,3,4,5,6,7]:
        block = 3
    elif period in [8,9]:
        block = 4
    else:
        print('error')
        return
    
    valence = group
    if group>12 and period<8:
        valence = group - 10
    if ele=='He':
        valence = 2
    
    # electro_negativity,covalent_radius(pm),first_ionization_energy*0.0103636(eV),electro_affinity(eV),molar_volume(cm^3/mol)
    property = {
    'H':[2.20,37,1312,0.754,22.413],'Li':[0.98,134,520.2,0.618,13.02],'Be':[1.57,90,899.5,-0.5,4.877],'B':[2.04,82,800.6,0.279,4.395],'C':[2.05,77,1086.5,1.262,5.29],'N':[3.04,75,1402.3,-0.07,22.413],'O':[3.44,73,1313.9,1.461,22.413],'F':[3.98,71,1681,3.401,11.202],'Na':[0.93,154,495.8,0.548,23.75],'Mg':[1.31,130,1450.7,-0.4,13.984],'Al':[1.61,118,577.5,0.433,9.99],'Si':[1.90,111,786.5,1.389,12.054],'P':[2.19,106,1011.8,0.746,16.991],'S':[2.58,102,999.6,2.077,15.53],'Cl':[3.16,99,1251.2,3.613,22.412],'K':[0.82,196,418.8,0.501,45.68],'Ca':[1.00,174,589.8,0.024,25.857],'Sc':[1.36,144,633.1,0.179,15.061],'Ti':[1.54,136,658.8,0.075,10.621],'V':[1.63,125,650.9,0.527,8.337],'Cr':[1.66,127,652.9,0.676,7.282],'Mn':[1.55,139,717.3,-0.5,7.354],'Fe':[1.83,125,762.5,0.153,7.092],'Co':[1.88,126,760.4,0.662,6.62],'Ni':[1.91,121,737.1,1.157,6.589],'Cu':[1.90,138,745.5,1.236,7.124],'Zn':[1.65,131,906.4,-0.6,9.161],'Ga':[1.81,126,578.8,0.301,11.809],'Ge':[2.01,122,762,1.232,13.646],'As':[2.18,119,947,0.805,12.95],'Se':[2.55,116,941,2.021,16.385],'Br':[2.96,114,1139.9,3.363,19.78],'Kr':[3.00,110,1350.8,-1,22.35],'Rb':[0.82,211,403,0.486,55.788],'Sr':[0.95,192,549.5,0.052,33.94],'Y':[1.22,162,600,0.311,19.881],'Zr':[1.33,148,640.1,0.433,14.011],'Nb':[1.6,137,652.1,0.917,10.841],'Mo':[2.16,145,684.3,0.747,9.333],'Tc':[1.9,156,702,0.55,8.522],'Ru':[2.2,126,710.2,1.046,8.171],'Rh':[2.28,135,719.7,1.143,8.265],'Pd':[2.20,131,804.4,0.562,8.851],'Ag':[1.93,153,731,1.304,10.283],'Cd':[1.69,148,867.8,-0.7,12.995],'In':[1.78,144,558.3,0.384,15.707],'Sn':[1.96,141,708.6,1.112,16.239],'Sb':[2.05,138,834,1.047,18.181],'Te':[2.1,135,869.3,1.971,20.449],'I':[2.66,133,1008.4,3.059,25.689],'Xe':[2.60,130,1170.4,-0.8,22.413],'Cs':[0.79,225,375.7,0.471,70.732],'Ba':[0.89,198,502.9,0.145,38.16],'Lu':[1.27,160,523.5,0.239,17.78],'Hf':[1.3,150,658.5,0.178,13.44],'Ta':[1.5,138,761,0.329,10.85],'W':[2.36,146,770,0.816,9.47],'Re':[1.9,159,760,0.06,8.86],'Os':[2.2,128,840,1.078,8.421],'Ir':[2.2,137,880,1.564,8.520],'Pt':[2.28,128,870,2.125,9.09],'Au':[2.54,144,890.1,2.308,10.21],'Hg':[2.0,149,1007.1,-0.5,14.09],'Tl':[1.62,148,589.4,0.320,17.24],'Pb':[1.87,147,715.6,0.356,18.27],'Bi':[2.02,146,703,0.942,21.31],'Po':[2.0,148,812.1,1.4,22.97],'At':[2.2,150,899.003,2.416,4.5],'Rn':[2.2,145,1037,-0.7,50.5],'Fr':[7.9],'Ra':[0.9],'Lr':[1.3,],'La':[1.1,169,538.1,0.557,22.386],'Ce':[1.12,166,534.4,0.60,20.947],'Pr':[1.13,165,527,0.109,20.8],'Nd':[1.14,164,533.1,0.097,20.576],'Sm':[1.17,162,544.5,0.162,19.98],'Gd':[1.2,162,593.4,0.212,19.903],'Tb':[1.1,160,565.8,0.131,19.336],'Dy':[1.22,158,573,0.015,19.004],'Ho':[1.23,160,581.0,0.338,18.753],'Er':[1.24,140,589.3,0.312,18.449],'Tm':[1.25,150,596.7,1.029,19.13],'Ac':[1.1,170,499,0.35,22.542],'Th':[1.3,169,587,0.607,19.792],'Pa':[1.5,160,568,0.55,15.18],'U':[1.38,158,597.6,0.315,11.589],'Np':[1.36,155,604.5,0.48,11.589],'Pu':[1.28,150,584.7,-0.5,12.29],'Am':[1.13],'Cm':[1.28],'Bk':[1.3],'Cf':[1.3],'Es':[1.3],'Fm':[1.3],'Md':[1.3],'Yb':[1.1,150,603.4,-0.02,24.84],'Pm':[1.13,162,540,0.129,20.23],'Ar':[0.5,100,1520.6,-1,22.413],'Eu':[1.2,160,547.1,0.116,19.98],'He':[0.5,35,2372.3,-0.5,22.413],'Ne':[0.5,72,2080.7,-1.2,22.42]
    }
    
    if len(property[ele])==1:
        print(ele,'error')
        return
    electro = property[ele][0] # 0.79-3.98
    coval = property[ele][1]  # 37-225
    firstio = property[ele][2] # 3.9-17.5
    elecaff = property[ele][3] # -1-3.61
    volume = property[ele][4] # 4.39-70.7
    
    
    res = [ change_to_N(electro,0.5,4,N),change_to_N(coval,35,230,N),change_to_N(firstio*0.0103636,3.8,25,N),change_to_N(elecaff,-1.2,3.7,N),change_to_N(volume,4.3,71,N) ]
    a1 = change_to_onehot(int(res[0]),N)
    a2 = change_to_onehot(int(res[1]),N)
    a3 = change_to_onehot(int(res[2]),N)
    a4 = change_to_onehot(int(res[3]),N)
    a5 = change_to_onehot(int(res[4]),N)
    a6 = change_to_onehot(group-1,18)
    a7 = change_to_onehot(period-1,9)
    a8 = change_to_onehot(block-1,4)
    a9 = change_to_onehot(valence-1,16)
    a = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9

    return a
    
    


def atom_quotient_complex(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
    ):
        
        
        edge_index,edge_dis,triangle_index,triangle_dis = get_complex(atoms,cutoff=cutoff,max_neighbors=max_neighbors)
        
        
        # cgcnn feature
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features='atomic_number'))
            #feat = get_cgcnn_value(s,30) # 47 + 5N
            sps_features.append(feat)
        
        sps_features = np.array(sps_features)
        
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        
        
        g = Data(x=node_features, edge_index=edge_index, edge_dis=edge_dis, triangle_index=triangle_index, triangle_dis=triangle_dis,label=0 )
        
        return g
        
        


def load_complexes(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
    #all_targets: list = [],
):
    """Construct quotient simplicial complex.

    """

    def atoms_to_complex(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return atom_quotient_complex(
            structure,
            neighbor_strategy=neighbor_strategy,
            cutoff=cutoff,
            #atom_features="atomic_number",
            atom_features='cgcnn',
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
            use_angle=use_angle,
        )
        
    #print('in--------------------------------------')
    #print(type(df['atoms']),atoms_to_graph)
    
    #complexes = df["atoms"].progress_apply(atoms_to_complex).values
    #complexes = df['atoms'].apply(atoms_to_complex).values
    complexes = df['atoms'].parallel_apply(atoms_to_complex).values
    return complexes



def get_train_val_test_loader(dataset,dataset_train,dataset_val,dataset_test,property,batch_size):
    # train
    df = pd.DataFrame(dataset_train)
    vals = df[property].values
    complexes = load_complexes(df,name=dataset)
    
    mean_train = np.mean(vals)
    std_train = np.std(vals)
    
    
    train_data = PygStructureDataset(
            df,
            complexes,
            target=property,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
    
    # val
    df = pd.DataFrame(dataset_val)
    vals = df[property].values
    complexes = load_complexes(df,name=dataset)
    val_data = PygStructureDataset(
            df,
            complexes,
            target=property,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
        
    # test
    df = pd.DataFrame(dataset_test)
    vals = df[property].values
    complexes = load_complexes(df,name=dataset)
    test_data = PygStructureDataset(
            df,
            complexes,
            target=property,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
    
    
    collate_fn = train_data.collate
    
    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        follow_batch = ['x','edge_dis']
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        follow_batch = ['x','edge_dis']
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        follow_batch = ['x','edge_dis']
    )
    
    prepare_batch = partial(train_loader.dataset.prepare_batch)
    return train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train