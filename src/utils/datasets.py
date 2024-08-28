#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training dataset
"""

import os
from os import listdir as ls
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join, isdir, isfile
import random
import torch.nn.functional as F



class TrainingDataset(Dataset):
    def __init__(self, root_dir, dict_indices, transform, mode='train', weighting="sqrt", daug_by_sum=False, get_link_length=False, generic_cml_index=False):
        """
        Args:
            root_dir (str): Directory with all the training/validation/test data.
            dict_indices (dict): Nested dictionary as described.
            mode (str): One of 'train', 'val_inter', 'val_intra', 'test_inter', 'test_intra'.
        """
        self.root_dir = root_dir
        self.dict_indices = dict_indices
        self.mode = mode
        if self.mode == 'train_fc_layers':
            self.dataset_subdir = 'train'
        else:
            self.dataset_subdir = self.mode
        self.samples = []
        self.weights = []
        self.lengths = []
        self.transform = transform
        self.daug_by_sum = daug_by_sum
        self.generic_cml_index = generic_cml_index
        
        if weighting == "sqrt":
            self.weighting_function = lambda x : 1 + 2 * np.sqrt(x)
        elif weighting == "linear":
            self.weighting_function = lambda x : 1 + 2 * x
            
        # Build the samples list and their weights
        for ls_id_str in sorted(ls(join(root_dir, self.dataset_subdir))):
            ls_id = int(ls_id_str)
            data = self.dict_indices[ls_id]
            
            for timestamp in sorted(ls(join(root_dir, self.dataset_subdir, ls_id_str))):
                self.samples.append((ls_id, timestamp))
                x = data['stats_antilope'][timestamp]['stats_precip_AB'][0]
                length = data['length']
                weight = self.weighting_function(x)
                self.weights.append(weight)
                self.lengths.append(length)
                
        print("nb of samples in ", mode, ':', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ls_id, timestamp = self.samples[idx]
        weight = self.weights[idx]
        link_length = self.lengths[idx]
    
        input_file = f'{ls_id}_{timestamp}_attenuations.pt'
        target_file = f'{ls_id}_{timestamp}_mean_precip.pt'
        try:
            inputs = torch.load(join(self.root_dir, self.dataset_subdir, str(ls_id), timestamp, input_file))
            targets = torch.load(join(self.root_dir, self.dataset_subdir, str(ls_id), timestamp, target_file))
        except :
            print("Failed loading file from " + join(self.root_dir, self.dataset_subdir, str(ls_id), timestamp))
            return None  
            


        if self.daug_by_sum  and (self.mode == 'train'):
            
            if (torch.rand(1) > 0.5):
                if (torch.rand(1) < 0.9) and self.generic_cml_index:
                    ls_id = -1
                idx2 = torch.randint(len(self.samples), (1,))
                link_length2 = self.lengths[idx2]
                ls_id2, timestamp2 = self.samples[idx2]
     
                input_file2 = f'{ls_id2}_{timestamp2}_attenuations.pt'
                target_file2 = f'{ls_id2}_{timestamp2}_mean_precip.pt'
                try:
                    inputs2 = torch.load(join(self.root_dir, self.dataset_subdir, str(ls_id2), timestamp2, input_file2))
                    targets2 = torch.load(join(self.root_dir, self.dataset_subdir, str(ls_id2), timestamp2, target_file2))

                    # transform
                    inputs, targets, start = self.transform(inputs, targets, ls_id)
                    inputs2, targets2, start2 = self.transform(inputs2, targets2, crop_start=start)  
                    
                    inputs += inputs2
                    
                    # print(torch.sum((targets < 0) * (targets2 < 0)))       
                    bad_targets = targets==-0.099
                    targets = (link_length * targets + link_length2 * targets2) / (link_length + link_length2)
                    targets[targets2==-10] = -10
                    targets[bad_targets] = -0.099
                    targets[targets2==-0.099] = -0.099
                    
                    link_length = link_length + link_length2 

                except :
                    print("Failed loading file from " + join(self.root_dir, self.mode, str(ls_id), timestamp))
                    pass
            else:
                if self.generic_cml_index and (torch.rand(1) > 0.9):
                    ls_id = -1
                inputs, targets, start = self.transform(inputs, targets, ls_id)

        else:
            if self.generic_cml_index and (((torch.rand(1) > 0.9) and self.mode == "train") or (self.mode in ["val_inter", "test_inter"])):
                ls_id = -1
            inputs, targets, start = self.transform(inputs, targets, ls_id)
                
        return timestamp, inputs, targets, link_length, ls_id
 
