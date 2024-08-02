from pre_train import train

dataset = 'jarvis' # 'jarvis' or 'mp'
epoch = 500
lr = 0.001  
batch_size = 64
weight_decay = 1e-05
property = [
            # jarvis
            'mbj_bandgap', 
            'optb88vdw_bandgap', 
            'formation_energy_peratom', 
            'optb88vdw_total_energy', 
            
            
            # material project
            'shear modulus',  
            'bulk modulus',   
            'gap pbe',   
            'e_form', 
            ]

train(dataset,epoch,lr,batch_size,weight_decay,property[0])
