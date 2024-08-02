import os
import torch
import random
import os.path as osp
import math
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import ignite
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.utils import convert_tensor
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from data import get_train_val_test
from quotientcomplex import get_train_val_test_loader
from model import QCformer






# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")








def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]



def train(dataset,epoch,learning_rate,batch_size,weight_decay,property,split_seed=123):
    os.makedirs(dataset + '-' + property  )
    
    dataset_train,dataset_val,dataset_test = get_train_val_test(dataset,property,split_seed)
 
    train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train = get_train_val_test_loader(dataset,dataset_train,dataset_val,dataset_test,property,batch_size)
    
    ignite.utils.manual_seed(split_seed)
    
    head_v = 4
    head_e = 2 
    layer_number = 5
    net = QCformer(64,64,head_v,head_e,layer_number )
    net.to(device)
    criterion = torch.nn.MSELoss()
    params = group_decay(net)
    optimizer = torch.optim.AdamW(params,lr=learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epoch,pct_start=0.3)
    
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=True,
    )
    val_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    test_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    
    '''
    # model checkpointing
    to_save = {"model": net,"optimizer": optimizer,"lr_scheduler": scheduler,"trainer": trainer}
    handler = Checkpoint(to_save, DiskSaver('saved/' + model_detail, create_dir=True, require_empty=False),
            n_saved=2, global_step_transform=lambda *_: trainer.state.epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
    
    # evaluate save
    to_save = {"model": net}
    handler = Checkpoint(to_save, DiskSaver('saved/' + model_detail, create_dir=True, require_empty=False),
            n_saved=2,filename_prefix='best',score_name="neg_mae",global_step_transform=lambda *_: trainer.state.epoch)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    '''
    
    
    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        "test": {m: [] for m in metrics.keys()},
    }
    
    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        val_evaluator.run(val_loader)
        vmetrics = val_evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)

        
        
        epoch_num = len(history["validation"][t_metric])
        if epoch_num % 10 == 0:
            # train
            train_evaluator.run(train_loader)
            tmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = tmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
            
        else:
            tmetrics = {}
            tmetrics['mae'] = -1
            #test_metrics = {}
            #test_metrics['mae'] = -1
        
        test_mae = 0
        if epoch_num==epoch:
            # test
            net.eval()
            with torch.no_grad():
                if epoch_num==epoch:
                    f = open(dataset + '-' + property + '/test_prediction.csv','w')
                    f.write("target,prediction\n")
                    targets = []
                    predictions = []
                    for dat in test_loader:
                        g,target = dat
                        target=target.to(device)
                        out_data = net(g.to(device))
                        pre1 = target.tolist()
                        pre2 = out_data.tolist()
                        for ii in range(len(pre1)):
                            f.write("%6f, %6f\n" % (pre1[ii], pre2[ii]))
                            #line = str(pre1[ii]) + "," + str(pre2[ii]) + "\n"
                            #f.write(line)
                        
                    f.close()
                for dat in test_loader:
                    g,target = dat
                    target=target.to(device)
                    out_data = net(g.to(device))
                    #print(target.shape,out_data.shape)
                    test_mae = test_mae + torch.abs(target-out_data).sum().data.item()
                
                    
            #print('Test MAE:',mae)


        # for metric in metrics.keys():
        #    history["train"][metric].append(tmetrics[metric])
        #    history["validation"][metric].append(vmetrics[metric])
        pbar = ProgressBar()
        if epoch_num<epoch:
            print('epoch:',epoch_num,f"Val_MAE: {vmetrics['mae']:.4f}",f"Train_MAE: {tmetrics['mae']:.4f}")
        else:
            print('epoch:',epoch_num,f"Val_MAE: {vmetrics['mae']:.4f}",f"Train_MAE: {tmetrics['mae']:.4f}","Test MAE",test_mae*std_train/len(dataset_test))
        #pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
        #pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
        

    trainer.run(train_loader, max_epochs=epoch)
    
    
    
    





        
    
