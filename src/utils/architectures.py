#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_archi function
"""
import torch

def load_archi(arch, nchannels, nclasses, size=64, dilation=1, atrous_rates=[6,12,18], fixed_cumul=False, additional_parameters=2, num_cmls=1000):
    if arch == "UNet_causal":
        from  src.utils.architectures_fcn import UNet_causal
        model = UNet_causal(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet":
        from  src.utils.architectures_fcn import UNet
        model = UNet(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_rescale":
        from  src.utils.architectures_fcn import UNet_rescale
        model = UNet_rescale(nchannels, nclasses, size, n_parameters=5000).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn":
        from  src.utils.architectures_fcn import UNet_causal_5mn
        model = UNet_causal_5mn(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous":
        from  src.utils.architectures_fcn import UNet_causal_5mn_atrous
        model = UNet_causal_5mn_atrous(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates, 
                                    fixed_cumul=fixed_cumul,
                                    additional_parameters=additional_parameters).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous_rescale":
        from  src.utils.architectures_fcn import UNet_causal_5mn_atrous_rescale
        model = UNet_causal_5mn_atrous_rescale(nchannels, nclasses, size,
                                    dilation=dilation, atrous_rates=atrous_rates, 
                                    fixed_cumul=fixed_cumul,
                                    additional_parameters=additional_parameters,
                                    num_cmls=num_cmls).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                                    
    if arch == "UNet_causal_5mn_atrous_complex_rescale":
        from  src.utils.architectures_fcn import UNet_causal_5mn_atrous_complex_rescale
        model = UNet_causal_5mn_atrous_complexe_rescale(nchannels, nclasses, size,
                                    dilation=dilation, atrous_rates=atrous_rates, 
                                    fixed_cumul=fixed_cumul,
                                    additional_parameters=additional_parameters,
                                    num_cmls=num_cmls).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model     



