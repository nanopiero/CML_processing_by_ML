#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAUG cmls
"""

from os import listdir as ls
from os.path import join, isdir, isfile
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random



class TimeSeriesTransform:
    def __init__(self, 
                 completion=False,
                 min_input_value=-2,
                 random_shifts=False, 
                 additive_noise=False, 
                 rescale_inputs=None,
                 rescale_targets="nan_padding", 
                 size_cropping=None,
                 step_cropping=1,
                 random_renorm=False,
                 daug_by_channel_inversion=False,
                 random_renorm_generic=False
                 ):
        
        self.completion = completion  # putain de virgule enlevée le 23/05 à 00:21
        self.min_input_value = min_input_value
        self.random_shifts = random_shifts
        self.additive_noise = additive_noise
        self.rescale_inputs = rescale_inputs
        self.rescale_targets = rescale_targets
        self.size_cropping = size_cropping
        self.step_cropping = step_cropping
        self.random_renorm = random_renorm
        self.daug_by_channel_inversion = daug_by_channel_inversion
        self.random_renorm_generic = random_renorm_generic
        
        if self.rescale_inputs:
            if 20 % self.rescale_inputs != 0:
                raise ValueError(f"20 must be divisible by {self.rescale_inputs}, but it is not.")
            self.spacing = 20 // self.rescale_inputs
        elif self.rescale_targets == 'diff':
            self.spacing = 5
        else:
            self.spacing = 20       

    def __call__(self, inputs, targets, crop_start=None, ind_cml=0):
        """
        crop_start permet de fixer le reste modulo 20 // step_cropping
        ne fonctionne que lorsque rescale_inputs = "nan_padding" 
        """
        start = 0
        
        if self.completion :  
            # fill missing values by the preceding ones
            pinputs = torch.cat((inputs[:2,1:], inputs[:2,[-1]]), dim=1)
            mask = inputs[:2,...] <= -50

            inputs[:2,...][mask] = pinputs[mask]
            # to reduce inputs res. at 1 mn (rescale_input=4) or 2 min 30 (rescale_input=10) or 5 min (rescale_input=20) resolution

            if self.rescale_inputs :
                inputs = self.reduce_resolution(inputs, kernel_size=self.rescale_inputs)   
                
            mask = inputs[:2,...] <= -50
            inputs[:2,...][mask] = 0
            
            # s'il en manque, on met à zéro
            inputs[:2,...] = torch.clamp(inputs[:2,...], min=self.min_input_value, max=50) / 10.

        else:
            # to reduce inputs res. at 1 mn (rescale_input=4) or 2 min 30 (rescale_input=10) or 5 min (rescale_input=20) resolution
            if self.rescale_inputs :
                inputs = self.reduce_resolution(inputs, kernel_size=self.rescale_inputs)                             
            inputs[:2,...] = torch.clamp(inputs[:2,...], min=self.min_input_value, max=50) / 10.  # premières tâches : min -2
            
        if self.random_shifts:
            inputs = self.apply_random_shifts(inputs)
            
        if self.additive_noise:
            inputs = self.add_noise(inputs)
        
        targets[0, ...] = targets[0,...].float() / 1000.
        targets[0, ...] = torch.clamp(targets[0,...], min=-0.5)


        if self.rescale_targets == "nan_padding":
            channels, seq_length = inputs.shape
            targets = self.space_targets(targets, inputs.shape)

            if self.size_cropping and self.size_cropping < seq_length:
                # general case
                if crop_start is None :
                    start = random.randint(0, (inputs.shape[1] - self.size_cropping) // self.step_cropping) * self.step_cropping
                    # print(start, self.size_cropping)
                # if daug_by_sum
                else: 
                    remainder = crop_start % self.spacing 
                    start = random.randint(0, (inputs.shape[1] - self.size_cropping) // self.spacing - 1) * self.spacing + remainder
                    # print(remainder, start, self.size_cropping)
                # testx :
                # start = 0 * self.step_cropping
                # print(start)
                inputs = inputs[:, start:start + self.size_cropping]
                targets = targets[:, start:start + self.size_cropping]
            
        elif self.rescale_targets == "diff":
            channels, seq_length = inputs.shape
            targets = self.space_targets(targets, (channels, 1 + seq_length // 4))
            if self.size_cropping and self.size_cropping < seq_length:
                start = random.randint(0, (inputs.shape[1] - self.size_cropping) // self.step_cropping) * self.step_cropping
                start_target = start // 4
                inputs = inputs[:, start:start + self.size_cropping]
                targets = targets[:, start_target:start_target + (self.size_cropping // 4)]

        if self.random_renorm:
            inputs = self.apply_random_renorm(inputs)
            
        if self.random_renorm_generic:
            if ind_cml != -1:
                inputs = self.apply_random_renorm(inputs)
            
        if self.daug_by_channel_inversion:
            if torch.rand(1) > 0.5:
                inputs = self.inverse_two_first_channels(inputs)
            
        return inputs, targets, start 

    
    def reduce_resolution(self, inputs, kernel_size=4):
        # The nan is encoded by -99. Other inputs values are greater than -9
        # hence kernel width could be up to 10 (more than 2 minutes)
        # in the experiment, kernel_size = 4 ( minute)
        # or 10 ( 2.5 minutes)
        
        # Handle -99 values (and other suspect values) for NaN representation
        nan_mask = inputs[:2,...] <= -50
        inputs[nan_mask] = -10000
        
        # Calculate padding to ensure alignment with round hours
        num_channels = inputs.shape[0]  # Number of channels
        pad_size = kernel_size - 1  # Example padding
        
        # Apply zero padding at the start of the time series
        inputs_padded = F.pad(inputs, (pad_size, 0), "constant", 0)
               
        # Define the convolution filter for each channel (grouped convolution)
        conv_filter = torch.ones((num_channels, 1, kernel_size), dtype=torch.float32) / kernel_size
        
        # Apply convolution with grouping
        scaled_inputs = F.conv1d(inputs_padded, conv_filter, stride=kernel_size, groups=num_channels)

        # Restore -99 values for NaNs
        scaled_inputs[scaled_inputs<=-50] = -99
        return scaled_inputs
        
    
    def apply_random_shifts(self, inputs):
        if torch.rand(1) < 0.5:
            N = inputs.shape[-1]
            a = np.arange(N)
            b = 100 + 400*np.random.rand(1)
            sin1 = np.sin( 2 * np.pi / (b * (1 + np.random.rand(1))) * a + np.random.rand(1))
            sin2 = np.sin( 2 * np.pi / (b * (1 + 2*np.random.rand(1))) * a + np.random.rand(1))
            sin3 = np.sin( 2 * np.pi / (b * (1 + 2*np.random.rand(1))) * a + np.random.rand(1))
            x = sin1 + sin2 + sin3
            y = 1*(x > 0.7) + 1*(x > 1.5) - 1*(x < -0.7) - 1*(x < -1.5)
            y[0:10] = 0
            y[-10:] = 0
            return inputs[..., a + y]
        else:
            return inputs

    def add_noise(self, inputs):
        if torch.rand(1) < 0.4:
            sigma1 = torch.abs(torch.randn(1) * 0.1)
            additional_sigmas = torch.abs(sigma1 + torch.randn(inputs.shape[0] - 1) * 0.02)
            sigmas = torch.cat([sigma1, additional_sigmas])
            noise = torch.randn_like(inputs) * sigmas.unsqueeze(1)
            inputs += noise
        
        return inputs

    def apply_random_renorm(self, inputs):
        if torch.rand(1) < 0.4:
            medians =  torch.median(inputs, dim=1).values.view(2,1)
            bias = medians * torch.rand(2,1) - 0.5*medians
            new_scale =  (0.7 + 0.3*torch.rand(2,1))
            inputs = (inputs - medians) * new_scale  + bias 
            
        return inputs

    
    def interpolate_target(self, target, target_size):
        # Reshape target to (batch_size, 1, seq_length) for interpolation
        target = target.unsqueeze(1)
        
        # Use linear interpolation
        upsampled_target = F.interpolate(target, size=target_size, mode='linear', align_corners=False)
        
        return upsampled_target.squeeze(1)
    
    
    
    def space_targets(self, targets, inputs_shape):
        # Assume input_shape is (batch_size, seq_length)
        _, seq_length = inputs_shape
        nc, target_length = targets.shape
        new_targets = -10 * torch.ones((nc, seq_length), device=targets.device)

        # Create an index tensor for the positions in new_targets
        indices = torch.arange(0, (target_length -1) * self.spacing + 1, self.spacing)
        new_targets[:, indices] = targets

        return new_targets


    def inverse_two_first_channels(self, inputs):
        inputs[[0,1],:] = inputs[[1,0],:]
        return inputs