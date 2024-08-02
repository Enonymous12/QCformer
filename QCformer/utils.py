import torch
from typing import Optional
import math
import numpy as np
from jarvis.core.specie import chem_data, get_node_attributes



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
    
    
    if ele not in property or len(property[ele])==1:
        return None
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
    


def get_attribute_lookup():
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features='cgcnn')
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features='cgcnn')
        #x = get_cgcnn_value(element,20) # 47 + 5N
        if x is not None:
            features[z, :] = x

    return features

FEATURE = get_attribute_lookup()
FEATURE = torch.from_numpy(FEATURE).to(dtype=torch.float32)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
FEATURE = FEATURE.to(device)


class RBFExpansion(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
        
        
class RBFExpansion_node(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(self):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return FEATURE[ distance.squeeze().long() ]
        
         
        
        
class RBFExpansion_edge(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 64,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        self.gamma1 = 1/0.01
        self.gamma2 = 1/0.1 
        self.gamma3 = 1

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        a1 = torch.exp(
            -self.gamma1 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a2 = torch.exp(
            -self.gamma2 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a3 = torch.exp(
            -self.gamma3 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a6 = FEATURE[ distance[:,0].long() ]
        a7 = FEATURE[ distance[:,1].long() ]
        return torch.cat([a6,a7,a1,a2,a3],dim=1)
        

class RBFExpansion_triangle(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 64,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        self.gamma1 = 1/0.01
        self.gamma2 = 1/0.1 
        self.gamma3 = 1

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        a1 = torch.exp(
            -self.gamma1 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a2 = torch.exp(
            -self.gamma2 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a3 = torch.exp(
            -self.gamma3 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a6 = FEATURE[ distance[:,0].long() ]
        a7 = FEATURE[ distance[:,1].long() ]
        a8 = FEATURE[ distance[:,2].long() ]
        return torch.cat([a6,a7,a8,a1,a2,a3],dim=1)
        
        
