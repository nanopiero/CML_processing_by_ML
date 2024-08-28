#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_archi function
"""


def load_archi(arch, nchannels, nclasses, size=64, dilation=1, atrous_rates=[6,12,18],fixed_cumul=False, additional_parameters=2):
    if arch == "UNet_causal":
        from ia.learning.dependencies.architectures_fcn import UNet_causal
        model = UNet_causal(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet":
        from ia.learning.dependencies.architectures_fcn import UNet
        model = UNet(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_rescale":
        from ia.learning.dependencies.architectures_fcn import UNet_rescale
        model = UNet_rescale(nchannels, nclasses, size, n_parameters=5000).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn":
        from ia.learning.dependencies.architectures_fcn import UNet_causal_5mn
        model = UNet_causal_5mn(nchannels, nclasses, size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous":
        from ia.learning.dependencies.architectures_fcn import UNet_causal_5mn_atrous
        model = UNet_causal_5mn_atrous(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates, 
                                       fixed_cumul=fixed_cumul,
                                       additional_parameters=additional_parameters).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous_rescale":
        from dependencies.architectures_fcn import UNet_causal_5mn_atrous_rescale
        model = UNet_causal_5mn_atrous_rescale(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates, 
                                               fixed_cumul=fixed_cumul,
                                               additional_parameters=5000).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous_complexe_rescale":
        from ia.learning.dependencies.architectures_fcn import UNet_causal_5mn_atrous_complexe_rescale
        model = UNet_causal_5mn_atrous_complexe_rescale(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates, 
                                               fixed_cumul=fixed_cumul,
                                               additional_parameters=5000).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_causal_5mn_atrous_complexe_rescale2":
        from ia.learning.dependencies.architectures_fcn import UNet_causal_5mn_atrous_complexe_rescale2
        model = UNet_causal_5mn_atrous_complexe_rescale2(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates, 
                                               fixed_cumul=fixed_cumul,
                                               additional_parameters=16).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_1mn_causal_5mn_atrous":
        from ia.learning.dependencies.architectures_fcn import UNet_1mn_causal_5mn_atrous
        model = UNet_1mn_causal_5mn_atrous(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_pseudo1mn_causal_5mn_atrous":
        from ia.learning.dependencies.architectures_fcn import UNet_pseudo1mn_causal_5mn_atrous
        model = UNet_pseudo1mn_causal_5mn_atrous(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "UNet_pseudo1mn_causal_5mn_atrous2":
        from ia.learning.dependencies.architectures_fcn import UNet_pseudo1mn_causal_5mn_atrous2
        model = UNet_pseudo1mn_causal_5mn_atrous2(nchannels, nclasses, size, dilation=dilation, atrous_rates=atrous_rates).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if arch == "DeepLab_causal_5mn":
        from ia.learning.dependencies.architectures_fcn import DeepLab_causal_5mn
        model = DeepLab_causal_5mn(
                    n_channels=nchannels,
                    n_classes=nclasses,
                    size=size,
                    n_blocks=[2, 2, 4, 3],
                    multi_grids=[1, 2, 4],
                    atrous_rates=atrous_rates
                )

    return model     



