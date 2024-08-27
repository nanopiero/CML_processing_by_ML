#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Créé le 06/05/2024 à partir de train/train_1GPU_newdaug.py (coca)

Appel:
mode interactif : 
python learning/preprocessing/train_1GPU_MAE_PNP.py fromscratch UNet_causal -bs 128 -ne 100 -pr 20240506_exp0
python learning/preprocessing/train_1GPU_MAE_PNP.py fromscratch UNet_causal -lr 0.003 -bs 128 -ne 100 -pr 20240506_exp1_ss30 -ss 30
python learning/preprocessing/train_1GPU_MAE_PNP.py fromscratch UNet_causal  -bs 128 -ne 100 -pr 20240506_exp2_1j -cs 5760
python learning/preprocessing/train_1GPU_MAE_PNP.py fromscratch UNet_causal  -bs 128 -ne 100 -pr 20240506_exp3_weightinglinear -w linear 
@author: lepetit
"""


##############################################################################
##############################################################################
#%% imports
##############################################################################
##############################################################################

from os.path import join, isdir, isfile
from os import listdir as ls
import argparse
import pickle
import numpy as np
import sys
import time
sys.path.append('/home/mdso/lepetitp/ppc/WEBCAMS/src/raincell')
import torch

from ia.learning.dependencies.transformations import TimeSeriesTransform
from ia.learning.dependencies.datasets import TrainingDataset, custom_collate_fn
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler

from ia.learning.dependencies.architectures import load_archi
from ia.learning.dependencies.cost_functions import CombinedLoss, CompleteLoss, compute_metrics, compute_confusion_matrix
import torch.optim as optim
from ia.learning.dependencies.scores import calculate_metrics, trailing_moving_average


print('disponibilite GPU: ')
print(torch.cuda.is_available())
print('-------------------')
print('nombre de gpus')
world_size = torch.cuda.device_count()
print(world_size)
print('----------')

##############################################################################
##############################################################################
#%% args
##############################################################################
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("load_params", 
            help="'pretrain', 'lastepo' ou 'bestloss' ")
parser.add_argument("archi")
parser.add_argument("-lr", "--learning_rate", type=float,
                    default=0.001)
parser.add_argument("-bs", "--batch_size", type=int,
                    default=64)
parser.add_argument("-ne", "--num_epochs", type=int,
                    default=50)
parser.add_argument("-pr", "--prefixe", type=str,
                    default="baseline_")
parser.add_argument("-ss", "--step_size", type=int,
                    default=10000)
parser.add_argument("-ga", "--gamma", type=float,
                    default=0.3)
parser.add_argument("-cs", "--crop_size", type=int,
                    default=2 * 24 * 60 * 4 + 5 * 60 * 4) # 2 days + 5heures
parser.add_argument("-cstep", "--crop_step", type=int,
                    default=1)
parser.add_argument("-w", "--weighting", type=str,
                    default="sqrt") # 2 days
parser.add_argument("-comp", "--completion", action='store_true')
parser.add_argument("-miv", "--min_input_value", type=int,
                    default=-2) #
parser.add_argument("-sm", "--size_model", type=int,
                    default=64) #
parser.add_argument("-rs", "--random_shifts", action='store_true')
parser.add_argument("-an", "--additive_noise", action='store_true')
parser.add_argument("-rr", "--random_renorm", action='store_true')
parser.add_argument("-rrg", "--random_renorm_generic", action='store_true')
parser.add_argument("-ri", "--rescale_inputs", type=int,
                    default=None)
parser.add_argument("-rt", "--rescale_targets", type=str,
                    default="nan_padding")
parser.add_argument("-lrc", "--long_receptive_field", action='store_true')
parser.add_argument("-ste", "--size_train_epoch", type=int, default=int(100_000 * 10_000 / (2 * 24 * 60 * 4 )))
parser.add_argument("-lbl", "--linear_balanced_loss", action='store_true')
parser.add_argument("-mcl", "--mode_complete_loss", type=str,
                    default=None)
parser.add_argument("-q", "--quantiles", type=int,
                    default=None)
parser.add_argument("-fc", "--fixed_cumul", action='store_true')
parser.add_argument("-ap", "--additional_parameters", type=int,
                    default=2)
parser.add_argument("-nvi", "--no_val_inter", action='store_true')
parser.add_argument("-dbci", "--daug_by_channel_inversion", action='store_true')
parser.add_argument("-dbs", "--daug_by_sum", action='store_true')
parser.add_argument("-tcs", "--target_cumsum", action='store_true')
parser.add_argument("-lor", "--learned_outputs_rescaling", action='store_true')
parser.add_argument("-locr", "--learned_outputs_complexe_rescaling", action='store_true')
parser.add_argument("-locr2", "--learned_outputs_complexe_rescaling2", action='store_true')
parser.add_argument("-gci", "--generic_cml_index", action='store_true')
parser.add_argument("-rni", "--renorm_inputs", action='store_true')
parser.add_argument("-il", "--input_lengths", action='store_true')

args = parser.parse_args()

print('args:')
print(args)


batch_size = args.batch_size 
num_epochs = args.num_epochs #180 #900
archi = args.archi  #'resnet50_imagenet_mtl' 
load = args.load_params
prefixe = args.prefixe
step_size = args.step_size
gamma = args.gamma
crop_size = args.crop_size
crop_step = args.crop_step
lr = args.learning_rate
weighting = args.weighting
completion = args.completion
min_input_value = args.min_input_value
random_shifts = args.random_shifts
additive_noise = args.additive_noise
random_renorm = args.random_renorm
random_renorm_generic = args.random_renorm_generic
rescale_inputs = args.rescale_inputs
size_model = args.size_model
long_receptive_field = args.long_receptive_field
rescale_targets = args.rescale_targets
size_train_epoch = args.size_train_epoch
if rescale_inputs:
    crop_size = crop_size // rescale_inputs
linear_balanced_loss = args.linear_balanced_loss
mode_complete_loss = args.mode_complete_loss
quantiles = args.quantiles
fixed_cumul = args.fixed_cumul
additional_parameters = args.additional_parameters
no_val_inter = args.no_val_inter
daug_by_channel_inversion = args.daug_by_channel_inversion
daug_by_sum = args.daug_by_sum
target_cumsum = args.target_cumsum
learned_outputs_rescaling = args.learned_outputs_rescaling
learned_outputs_complexe_rescaling = args.learned_outputs_complexe_rescaling
learned_outputs_complexe_rescaling2 = args.learned_outputs_complexe_rescaling2
renorm_inputs = args.renorm_inputs
input_lengths = args.input_lengths
generic_cml_index = args.generic_cml_index




##############################################################################
##############################################################################
#%% config
##############################################################################
##############################################################################

torch.manual_seed(18)

# compl. archi
nchannels = 2
if input_lengths:
    nchannels = 2
nclasses = 3
if mode_complete_loss in ['CE_MSE_MAE', 'CE_maskedMSE_maskedMAE']:
    nclasses = 4 # 2 segmentation, 1 MSE, 1 MAE
if mode_complete_loss in ['CE_MSE_Q', 'CE_maskedMSE_maskedQ']:
    nclasses = 3 + quantiles - 1  
if mode_complete_loss in ['CE_MSE_MSE1h', 'CE_MSE_MSEsum5min']:
    nclasses = 3 + 1

if learned_outputs_complexe_rescaling2:
    nclasses = nclasses + 4
    
# config loaders
num_workers = 4
# size_val = 5000


# GPU
k = 0
device = torch.device("cuda:" + str(k))

# Nomenc.
if 'exp19' in prefixe:
    model_name = prefixe +  '_' + archi \
               + "_lr" + str(int(1000*lr)) \
               + "step_size" + "30" + '_gamma' + str(int(1000*gamma)) + '_crop_size' + str(int(crop_size)) 
else:
    model_name = prefixe +  '_' + archi \
               + "_lr" + str(int(1000*lr)) \
               + "step_size" + str(step_size) + '_gamma' + str(int(1000*gamma)) + '_crop_size' + str(int(crop_size)) + '_crop_step' + str(int(crop_step))


print("model_name :" , model_name)

dir_data = '/scratch/mdso/lepetitp/ppc/RAINCELL/datasets/debiasing_20240814'
dir_models = "/scratch/mdso/lepetitp/ppc/RAINCELL/models/models_debiasing_20240814"

# Checkpoints paths
save_every = 1
PATH_bestloss_checkpoint = join(dir_models, model_name + "_bm.checkpoint")
PATH_bestloss_inter_checkpoint = join(dir_models, model_name + "_bmi.checkpoint")
PATH_lastepo_checkpoint = join(dir_models, model_name + "_le.checkpoint")
PATH_bestloss_intra_mse_checkpoint = join(dir_models, model_name + "_bm_mse.checkpoint")
PATH_bestloss_intra_segmentation_checkpoint = join(dir_models, model_name + "_bm_segmentation.checkpoint")


##############################################################################
##############################################################################
#%% load metadata
##############################################################################
##############################################################################

# dictionnaire des md de l'ensemble des indices de CMLs:

with open('/scratch/mdso/lepetitp/ppc/RAINCELL/METADATA/dict_indices_290424.pickle', 'rb') as file:
    dict_indices = pickle.load(file)
    
# get the links for which stats (used for sampling) are available
list_getstats = [i for i in dict_indices if dict_indices[i].get('stats_antilope') is not None]
print(len(list_getstats))
list_getstats_hydre = set()
for i in list_getstats:
    for ts in dict_indices[i]['stats_antilope']:
        if dict_indices[i]['stats_antilope'][ts].get('stats_hydre') is not None:
            list_getstats_hydre |= {i}
    
            
print(len(list_getstats_hydre))
print(len([i for i in dict_indices]))

# del keys with missing stats:
keys_to_del = [i for i in dict_indices if dict_indices[i].get('stats_antilope') is None]
print('nb of keys to del: ', len(keys_to_del))
for k in keys_to_del:
    del dict_indices[k]
print(len([i for i in dict_indices]))   
    
    
##############################################################################
##############################################################################
#%% datasets and dataloaders
##############################################################################
##############################################################################



transforms = {}

transforms['train'] =  TimeSeriesTransform(completion=completion,
                                            min_input_value=min_input_value, 
                                            random_shifts=random_shifts, 
                                            additive_noise=additive_noise, 
                                            rescale_inputs=rescale_inputs,
                                            rescale_targets=rescale_targets, 
                                            size_cropping=crop_size,
                                            step_cropping=crop_step,
                                            random_renorm=random_renorm,
                                            daug_by_channel_inversion=daug_by_channel_inversion,
                                            random_renorm_generic=random_renorm_generic
                                           ) 

transforms['val_intra'] = TimeSeriesTransform(completion=completion,
                                                min_input_value=min_input_value, 
                                                random_shifts=False, 
                                                additive_noise=False, 
                                                rescale_inputs=rescale_inputs,
                                                rescale_targets=rescale_targets, 
                                                size_cropping=None) 

transforms['val_inter'] = TimeSeriesTransform(completion=completion,
                                                min_input_value=min_input_value, 
                                                random_shifts=False, 
                                                additive_noise=False, 
                                                rescale_inputs=rescale_inputs,
                                                rescale_targets=rescale_targets, 
                                                size_cropping=None)

# Create datasets using a dictionary 
if no_val_inter :
    steps = ['train', 'val_intra']
else:
    steps = ['train', 'val_intra', 'val_inter']    

dict_daug_by_sum = {'train':daug_by_sum, 'val_intra':False, 'val_inter':False}
datasets = {step: TrainingDataset(dir_data, dict_indices, transforms[step], mode=step, weighting=weighting, daug_by_sum=dict_daug_by_sum[step], 
                                  generic_cml_index=generic_cml_index) for step in steps}



# Create  dataloaders using dictionaries
samplers = {}
dataloaders = {}


# training case : 100000 x (10.000 / crop_size)
nb_steps_sampler_train = size_train_epoch 

sampler_train = WeightedRandomSampler(datasets['train'].weights, nb_steps_sampler_train, replacement=True)

for step in datasets:
    if step == 'train':
        dataloaders[step] = DataLoader(datasets[step],
                                       batch_size=batch_size,
                                       sampler=sampler_train,
                                       num_workers=num_workers
                                      )
    else:
        dataloaders[step] = DataLoader(datasets[step],
                                       batch_size=batch_size,
                                       sampler=SequentialSampler(datasets[step]),
                                       num_workers=num_workers)



# Access specific dataloader, for example:
print("len dataloader train : ", len(sampler_train), " nb steps sampler train : ", nb_steps_sampler_train)
print("len dataloader val intra : ", len(SequentialSampler(datasets['val_intra'])))
if not no_val_inter:
    print("len dataloader val inter : ", len(SequentialSampler(datasets['val_inter'])))


##############################################################################
##############################################################################
#%% model & optim
##############################################################################
##############################################################################

if long_receptive_field :
    dilation=2
    atrous_rates=[6, 12, 18, 24, 30, 36, 42]
    model = load_archi(archi, nchannels, nclasses, size=size_model, dilation=dilation, atrous_rates=atrous_rates, 
                       fixed_cumul=fixed_cumul, additional_parameters=additional_parameters)

else:
    model = load_archi(archi, nchannels, nclasses, size=size_model, atrous_rates=[6, 12, 18],
                       fixed_cumul=fixed_cumul, additional_parameters=additional_parameters)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
if mode_complete_loss is not None:
    criterion = CompleteLoss(mode=mode_complete_loss, quantiles=quantiles)
else :
    criterion = CombinedLoss(linear_balanced_loss = linear_balanced_loss)
    
# criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
# scheduler_start = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., total_iters=5)


##############################################################################
##############################################################################
#%% load older weigths
##############################################################################
##############################################################################

last_epoch = -1
best_loss = [float('inf'), float('inf'), float('inf'), float('inf')]  # Initialize best validation loss to a very high value
train_losses = []
val_intra_losses = []
val_inter_losses = []



# if needed, init stats, best weights from checkpoint 
print('fromscratch or checkpoint :')
print(load)
if load != 'fromscratch':
    if load == 'lastepo': 
        PATH_weights = PATH_lastepo_checkpoint
    elif load == 'bestloss':
        PATH_weights = PATH_bestloss_checkpoint
    elif load == 'bestloss_inter':
        PATH_weights = PATH_bestloss_inter_checkpoint
    else:
        raise Exception('argument load_params mal renseigné: fromscratch, bestloss or lastepo')
    checkpoint = torch.load(PATH_weights, \
                            map_location=device)
    last_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_intra_losses = checkpoint['val_intra_losses']
    val_inter_losses = checkpoint['val_inter_losses']
    best_loss = checkpoint['best_loss']
    if not isinstance(best_loss, list):
        print(best_loss)
        best_loss = [best_loss, float('inf')]
    if len(best_loss) == 2: 
        best_loss += [float('inf'), float('inf')]
        print('inf inf appended to best loss', best_loss)

    model_weights = checkpoint['model']
    optimizer_state_dict = checkpoint['optimizer']
    scheduler_state_dict = checkpoint['scheduler']
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)
    # scheduler_start_state_dict = checkpoint['scheduler_start']
    # scheduler_start.load_state_dict(scheduler_start_state_dict)
    
#     print_gpu_state(rank)

    del checkpoint, model_weights, \
        optimizer_state_dict, \
        scheduler_state_dict

torch.cuda.empty_cache()






##############################################################################
##############################################################################
#%% training loop
##############################################################################
##############################################################################

since = time.time()
torch.autograd.set_detect_anomaly(True)

for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs ):
    print('Epoch {} - [{},{}]'.format(epoch, last_epoch + 1, last_epoch + num_epochs))
    print('-' * 10)
    

    # Training phase
    model.train()
    
    running_MAE_loss = 0.0
    running_MSE_loss = 0.0
    running_segmentation_loss = 0.0
    train_confusion_matrix = np.zeros((2, 2), dtype=int)
    # t = time.time()
    for batch_idx, (timestamps, noisy_series, reference, lengths, ids) in enumerate(dataloaders['train']):
        # print('batch load ', time.time() - t)
        
        
        # print(batch_idx)
        
        inputs, targets = noisy_series.to(device), reference.to(device)

        # if renorm_inputs: # (écrit le 15/07/2024 mais pas utilisé)
        #     lengths = lengths.to(device) / 15000.
        #     inputs /= lengths.view(inputs.shape[0],1,1)



        # t1 = time.time()
        if input_lengths:
            lengths = lengths.to(device) / 50_000.
            inputs = torch.cat([inputs, lengths.view(inputs.shape[0],1,1).repeat(1,1,inputs.shape[2])], dim=1)
            print(inputs.shape, lengths)
            
        if learned_outputs_rescaling:
            outputs = model(inputs, indices=ids.to(device))
            outputs, p = outputs
            outputs[:,2:,:] *= p[5:].view(outputs.shape[0],1,1)
            
        elif learned_outputs_complexe_rescaling or learned_outputs_complexe_rescaling2:
            outputs = model(inputs, ids.to(device))
        else:
            outputs = model(inputs)
        
        if target_cumsum:
            lengths = lengths.to(device) / 1000.
            outputs[:,2:,:] /= lengths.view(outputs.shape[0],1,1)
            
        # Forward pass
        optimizer.zero_grad()           
        # print('forward ', time.time() - t1)
        # t2 = time.time()
        MAE_loss, segmentation_loss, loss, batch_cm, MSE_loss = criterion(model.p[:5], outputs, targets, display_other_stats=False)[:5]  # Ensure that the arguments are correctly placed
        # print('loss ', time.time() - t2)
        # Backward and optimize
        # t3 = time.time()
        loss.backward()
        # print('backward ', time.time() - t3)
        # t4 = time.time()
        optimizer.step()
        # print('optim step ', time.time() - t4)
        
        del inputs, targets, outputs, loss
        torch.cuda.empty_cache()  
        
        running_MAE_loss += MAE_loss.item()
        running_MSE_loss += MSE_loss.item()
        running_segmentation_loss += segmentation_loss.item()
        train_confusion_matrix += batch_cm
        # print('whole step ', time.time() - t)
        # t = time.time()
        
    # Calculating average training loss
    train_MAE_loss = running_MAE_loss / len(dataloaders['train'])
    train_MSE_loss = running_MSE_loss / len(dataloaders['train'])
    train_segmentation_loss = running_segmentation_loss / len(dataloaders['train'])
    train_losses.append((epoch, train_MAE_loss, train_segmentation_loss, train_confusion_matrix, train_MSE_loss))
    print(f'Training, MAE Loss: {train_MAE_loss:.4f}, MSE Loss: {train_MSE_loss:.4f}, Segmentation Loss:{train_segmentation_loss:.4f}' )
    print("Train Confusion Matrix:")
    print(train_confusion_matrix)
    accuracy, csi, sensitivity, specificity, false_alarm_ratio = compute_metrics(train_confusion_matrix)
    print(f'Accuracy: {accuracy:.4f}, CSI: {csi:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, False Alarm Ratio: {false_alarm_ratio:.4f}')
    print('\n')   

    # Validation phases
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():

        # Validation intra
        running_MAE_loss = 0.0
        running_MSE_loss_preds = 0.0
        running_MSE_loss_gt = 0.0
        running_MSE_loss_tot = 0.0
        running_segmentation_loss = 0.0
        val_intra_confusion_matrix = np.zeros((2, 2), dtype=int)
        
        for timestamps, noisy_series, reference, lengths, ids in dataloaders['val_intra']:
            inputs, targets = noisy_series.to(device), reference.to(device)
            
            # if renorm_inputs:
            #     lengths = lengths.to(device) / 15000.
            #     inputs /= lengths.view(inputs.shape[0],1,1)
                
            # Forward pass
            # outputs = model(inputs[:, :, 1:]) before the 18/05. But non 5min-causal
            if learned_outputs_rescaling:
                outputs = model(inputs[:, :, :-1], indices=ids.to(device))
                outputs, p = outputs
                outputs[:,2:,:] *= p[5:].view(outputs.shape[0],1,1)
            elif learned_outputs_complexe_rescaling or learned_outputs_complexe_rescaling2:
                outputs = model(inputs[:, :, :-1], ids.to(device))
            else:
                outputs = model(inputs[:, :, :-1])

  
            if target_cumsum:
                lengths = lengths.to(device) / 1000.
                outputs[:,2:,:] /= lengths.view(outputs.shape[0],1,1)
                
            MAE_loss, segmentation_loss, loss, batch_cm, MSE_loss_preds, MSE_loss_gt, MSE_loss_tot  = criterion(model.p, outputs, targets[:, :, :-1], display_other_stats=True)[:7]    
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()    
            running_MAE_loss += MAE_loss.item()
            running_MSE_loss_preds += MSE_loss_preds.item()
            running_MSE_loss_gt += MSE_loss_gt.item()
            running_MSE_loss_tot += MSE_loss_tot.item()
            running_segmentation_loss += segmentation_loss.item()
            val_intra_confusion_matrix += batch_cm
            
        
        # Calculating average training loss
        val_intra_MAE_loss = running_MAE_loss / len(dataloaders['val_intra'])
        val_intra_MSE_loss_preds = running_MSE_loss_preds / len(dataloaders['val_intra'])
        val_intra_MSE_loss_gt = running_MSE_loss_gt / len(dataloaders['val_intra'])
        val_intra_MSE_loss_tot = running_MSE_loss_tot / len(dataloaders['val_intra'])
        val_intra_segmentation_loss = running_segmentation_loss / len(dataloaders['val_intra'])
        val_intra_losses.append((epoch, val_intra_MAE_loss, val_intra_segmentation_loss, val_intra_confusion_matrix, val_intra_MSE_loss_preds, val_intra_MSE_loss_gt, val_intra_MSE_loss_tot))

        print(f'Val Intra, MAE Loss: {val_intra_MAE_loss:.4f}, MSE Loss preds: {val_intra_MSE_loss_preds:.4f}, MSE Loss gt: {val_intra_MSE_loss_gt:.4f}, MSE Loss tot: {val_intra_MSE_loss_tot:.4f}, Segmentation Loss:{val_intra_segmentation_loss:.4f}' )
        print("val intra Confusion Matrix:")
        print(val_intra_confusion_matrix)
        accuracy, csi, sensitivity, specificity, false_alarm_ratio = compute_metrics(val_intra_confusion_matrix)
        print(f'Accuracy: {accuracy:.4f}, CSI: {csi:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, False Alarm Ratio: {false_alarm_ratio:.4f}')
        print('\n')
        
        # Validation inter
        if not no_val_inter:
            running_MAE_loss = 0.0
            running_MSE_loss_preds = 0.0
            running_MSE_loss_gt = 0.0
            running_MSE_loss_tot = 0.0
            val_inter_confusion_matrix = np.zeros((2, 2), dtype=int)
            
            for timestamps, noisy_series, reference, lengths, ids in dataloaders['val_inter']:
                inputs, targets = noisy_series.to(device), reference.to(device)
                
                # if renorm_inputs:
                #     lengths = lengths.to(device) / 15000.
                #     inputs /= lengths.view(outputs.shape[0],1,1)
                    
                # Forward pass
                if learned_outputs_rescaling:
                    outputs = model(inputs[:, :, :-1], indices=ids.to(device))
                    outputs, p = outputs
                    outputs[:,2:,:] *= p[5:].view(outputs.shape[0],1,1)
                elif learned_outputs_complexe_rescaling or learned_outputs_complexe_rescaling2:
                    outputs = model(inputs[:, :, :-1], ids.to(device))
                else:
                    outputs = model(inputs[:, :, :-1])
                
                if target_cumsum:
                    lengths = lengths.to(device) / 1000.
                    outputs[:,2:,:] /= lengths.view(outputs.shape[0],1,1)
                    
                    
                MAE_loss, segmentation_loss, loss, batch_cm,  MSE_loss_preds, MSE_loss_gt, MSE_loss_tot = criterion(model.p, outputs, targets[:, :, :-1], display_other_stats=True)[:7]  
                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()
                running_MAE_loss += MAE_loss.item()
                running_MSE_loss_preds += MSE_loss_preds.item()
                running_MSE_loss_gt += MSE_loss_gt.item()
                running_MSE_loss_tot += MSE_loss_tot.item()
                running_segmentation_loss += segmentation_loss.item()
                val_inter_confusion_matrix += batch_cm

    
                
            # Calculating average training loss
            val_inter_MAE_loss = running_MAE_loss / len(dataloaders['val_inter'])
            val_inter_MSE_loss_preds = running_MSE_loss_preds / len(dataloaders['val_inter'])
            val_inter_MSE_loss_gt = running_MSE_loss_gt / len(dataloaders['val_inter'])
            val_inter_MSE_loss_tot = running_MSE_loss_tot / len(dataloaders['val_inter'])
            val_inter_segmentation_loss = running_segmentation_loss / len(dataloaders['val_inter'])
            val_inter_losses.append((epoch, val_inter_MAE_loss, val_inter_segmentation_loss, val_inter_confusion_matrix, val_inter_MSE_loss_preds, val_inter_MSE_loss_gt, val_inter_MSE_loss_tot))
            print(f'Val Inter, MAE Loss: {val_inter_MAE_loss:.4f}, MSE Loss preds: {val_inter_MSE_loss_preds:.4f}, MSE Loss gt: {val_inter_MSE_loss_gt:.4f}, MSE Loss tot: {val_inter_MSE_loss_tot:.4f}, Segmentation Loss:{val_inter_segmentation_loss:.4f}' )
            print("val inter Confusion Matrix:")
            print(val_inter_confusion_matrix)
            accuracy, csi, sensitivity, specificity, false_alarm_ratio = compute_metrics(val_inter_confusion_matrix)
            print(f'Accuracy: {accuracy:.4f}, CSI: {csi:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, False Alarm Ratio: {false_alarm_ratio:.4f}')
            print('\n')

    
    
    scheduler.step()

    # savings
    
    
    if (epoch % save_every == 0 or \
        epoch == last_epoch + num_epochs):
        print("saving step - lastepo")
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_intra_losses' : val_intra_losses,
            'val_inter_losses' : val_inter_losses
            }
        torch.save(checkpoint, PATH_lastepo_checkpoint)  
           
             
    #critère1 de sauvegarde : meilleure en Mrégression intra (MAE)
    if val_intra_MAE_loss < best_loss[0]:
        print("saving step - bestloss")
        best_loss[0] = val_intra_MAE_loss
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_intra_losses' : val_intra_losses,
            'val_inter_losses' : val_inter_losses
        }

        torch.save(checkpoint, PATH_bestloss_checkpoint)
        print(f"Model saved: Improved regression on val. intra to {best_loss[0]:.4f}")

    #critère2 de sauvegarde : meilleure en Mrégression inter (MAE)
    if val_inter_MAE_loss < best_loss[1]:
        print("saving step - bestloss")
        best_loss[1] = val_inter_MAE_loss
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_intra_losses' : val_intra_losses,
            'val_inter_losses' : val_inter_losses
        }

        torch.save(checkpoint, PATH_bestloss_inter_checkpoint)
        print(f"Model saved: Improved regression on val. inter to {best_loss[1]:.4f}")

    #critère3 de sauvegarde : meilleure en MSE intra 
    if val_intra_MSE_loss_preds < best_loss[2]:
        print("saving step - bestloss")
        best_loss[2] = val_intra_MSE_loss_preds
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_intra_losses' : val_intra_losses,
            'val_inter_losses' : val_inter_losses
        }

        torch.save(checkpoint, PATH_bestloss_intra_mse_checkpoint )
        print(f"Model saved: Improved regression on mse val. intra to {best_loss[2]:.4f}")
        
    if val_intra_segmentation_loss < best_loss[3]:
        print("saving step - bestloss")
        best_loss[3] = val_intra_segmentation_loss
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses,
            'val_intra_losses' : val_intra_losses,
            'val_inter_losses' : val_inter_losses
        }

        torch.save(checkpoint, PATH_bestloss_intra_segmentation_checkpoint)
        print(f"Model saved: Improved segmentation val. intra to {best_loss[3]:.4f}")        


           
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val acc: {:4f}'.format(best_loss[0]))



##############################################################################
##############################################################################
#%% save stats 
##############################################################################
##############################################################################

