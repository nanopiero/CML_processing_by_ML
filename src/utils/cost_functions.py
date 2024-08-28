#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main cost function for CMLs processsing
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class CompleteLoss(nn.Module):
    def __init__(self, mode='CE_MSE', quantiles=None):
        super(CompleteLoss, self).__init__()
        self.criterion_segmentation = nn.CrossEntropyLoss(ignore_index=-100)
        self.mode = mode
        self.quantiles = quantiles
        if  quantiles is not None:
            self.quantile_loss = QuantileLoss1d(quantiles)
            
    def cumsum1h(self, x, mask):
        # Define the window size
        window_size = 12
        
        # Prepare convolutional weights for summing elements
        conv_weights = torch.ones((1, 1, window_size), dtype=torch.float32, device=x.device)
    
        # Compute sum using convolution, no padding
        sums = F.conv1d(x.unsqueeze(0), conv_weights)
    
        # Process the mask similarly to check if all are True
        mask_float = mask.float().unsqueeze(0)  # Convert mask to float and add channel dimension
        mask_sums = F.conv1d(mask_float, conv_weights)
    
        # Check where the mask sums equal the window size (i.e., all True in the window)
        valid_mask = (mask_sums == window_size).squeeze(1)  # Remove the unnecessary channel dimension
    
        return sums.squeeze(1)[valid_mask], \
               torch.cat([torch.zeros((1,window_size-1), device=x.device) == 1, valid_mask], dim=1)
    
    def forward(self, p, outputs, targets, display_other_stats=False):
        # Ensure targets have an extra dimension for broadcasting if needed
        # if outputs.dim() > targets.dim():
        #     targets = targets.unsqueeze(1) 


        mask = (targets != -10) 
        bs = targets.shape[0]
        
        valid_segmentation_p0 = outputs[:, 0, :][mask.squeeze(1)].view(bs, 1, -1)
        valid_segmentation_p1 = outputs[:, 1, :][mask.squeeze(1)].view(bs, 1, -1)
        valid_segmentation_outputs = torch.cat([valid_segmentation_p0, valid_segmentation_p1], dim=1)
        valid_segmentation_targets = (targets > 0).float()[mask].view(bs,-1).long()

        valid_intensity_targets = targets[mask]
        
        # when antilope = -99
        mask2 = valid_intensity_targets != -0.099
        valid_segmentation_targets[~mask2.view(bs,-1)] = -100 # ignore_index of CrossEntropyLoss
        segmentation_loss = self.criterion_segmentation(valid_segmentation_outputs, valid_segmentation_targets).mean()
        with torch.no_grad():
            cm = compute_confusion_matrix(valid_segmentation_outputs,
                                                 valid_segmentation_targets,
                                                 num_classes=2,
                                                 ignore_margin=48)  
            
        if self.mode == 'CE_MSE':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            MSE = ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask2].mean() 
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE  
            loss += torch.log(1 + p[0]**2 + p[1]**2)
            
            with torch.no_grad():
                MAE = (torch.abs(valid_intensity_MSE_outputs - valid_intensity_targets))[mask2].mean()

            if display_other_stats:
                with torch.no_grad():
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2

                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()

                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()
                    
                return MAE.detach(),  segmentation_loss, loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:        
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()

        elif self.mode == 'CE_maskedMSE':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            mask3 = valid_intensity_targets > 0
            MSE =  ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask3].mean() 
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE  
            loss += torch.log(1 + p[0]**2 + p[1]**2)
            

            if display_other_stats:
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach()))
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2

                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()
                    MAE[preds0] = 0
                    MAE = MAE[mask2].mean()
                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()
                    
                return MAE, segmentation_loss, loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:        
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MSE_outputs - valid_intensity_targets))
                    MSE = (valid_intensity_MSE_outputs - valid_intensity_targets)**2
                    preds = valid_segmentation_outputs.argmax(dim=1).flatten()
                    # print(preds.shape, MAE.shape)
                    MAE[preds == 0] = 0
                    MSE[preds == 0] = 0 
                    MAE = MAE[mask2].mean()
                    MSE = MSE[mask2].mean()
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()

        elif self.mode == 'CE_weightedmaskedMSE':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            mask3 = valid_intensity_targets > 0
            MSE = (valid_intensity_MSE_outputs - valid_intensity_targets)**2
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * (1 + (valid_intensity_targets.detach()) * MSE)[mask3].mean()   
            loss += torch.log(1 + p[0]**2 + p[1]**2)

            if display_other_stats:
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach()))
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2

                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()
                    MAE[preds0] = 0
                    MAE = MAE[mask2].mean()
                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()
                    
                return MAE, segmentation_loss.detach(), loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:   
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MSE_outputs - valid_intensity_targets))
                    preds = valid_segmentation_outputs.argmax(dim=1).flatten()
                    # print(preds.shape, MAE.shape)
                    MAE[preds == 0] = 0
                    MSE[preds == 0] = 0 
                    MAE = MAE[mask2].mean()
                    MSE = MSE[mask2].mean()
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()
        
        elif self.mode == 'CE_MSE_MSEsum5min':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            MSE = ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask2].mean() 
            valid_intensity_MSE1h_outputs, _ =  self.cumsum1h(outputs[:, 3, :][mask.squeeze(1)], mask2)
            with torch.no_grad():
                valid_intensity_targets1h, _ = self.cumsum1h(valid_intensity_targets, mask2)
            MSE1h = ((valid_intensity_MSE1h_outputs - valid_intensity_targets1h)**2).mean() 
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * MSE1h  
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)
            
            with torch.no_grad():
                MAE = (torch.abs(valid_intensity_MSE_outputs - valid_intensity_targets))[mask2].mean()
            return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()

        elif self.mode == 'CE_MSE_MSE1h':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            MSE = ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask2].mean() 
            with torch.no_grad():
                valid_intensity_targets1h, mask3 = self.cumsum1h(valid_intensity_targets, mask2)
            valid_intensity_MSE1h_outputs =  (outputs[:, 3, :][mask.squeeze(1)])[mask3.squeeze(0)]
            MSE1h = ((valid_intensity_MSE1h_outputs - valid_intensity_targets1h)**2).mean() 
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * MSE1h  
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)
            
            with torch.no_grad():
                MAE = (torch.abs(valid_intensity_MSE_outputs - valid_intensity_targets))[mask2].mean()
            return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()
        
        
        elif self.mode == 'CE_MSE_MAE': 
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            MSE = ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask2].mean() 
            valid_intensity_MAE_outputs = outputs[:, 3, :][mask.squeeze(1)]
            MAE = (torch.abs(valid_intensity_MAE_outputs - valid_intensity_targets))[mask2].mean()
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * MAE 
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)

            if display_other_stats:
                with torch.no_grad():
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2
                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()
                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()                    
                return MAE.detach(), segmentation_loss.detach(), loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:   
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()

        elif self.mode == 'CE_maskedMSE_maskedMAE':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            valid_intensity_MAE_outputs = outputs[:, 3, :][mask.squeeze(1)]
            mask3 = valid_intensity_targets > 0
            MSE =  ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask3].mean() 
            MAE = (torch.abs(valid_intensity_MAE_outputs - valid_intensity_targets))[mask3].mean()
            
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * MAE 
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)

            if display_other_stats:
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach()))
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2
                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()
                    MAE[preds0] = 0
                    MAE = MAE[mask2].mean()
                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()                    
                return MAE, segmentation_loss.detach(), loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:   
                with torch.no_grad():
                    MAE = (torch.abs(valid_intensity_MAE_outputs - valid_intensity_targets))
                    MSE = (valid_intensity_MSE_outputs - valid_intensity_targets)**2
                    preds = valid_segmentation_outputs.argmax(dim=1).flatten()
                    # print(preds.shape, MAE.shape)
                    MAE[preds == 0] = 0
                    MSE[preds == 0] = 0 
                    MAE = MAE[mask2].mean()
                    MSE = MSE[mask2].mean()
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()
        
        elif self.mode == 'CE_MSE_Q':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            MSE = ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask2].mean() 
            
            valid_intensity_quantile_outputs = outputs[:, 3:, :][mask.expand(bs, outputs.shape[1] - 3, -1)].view(bs, outputs.shape[1] - 3, -1)

            with torch.no_grad():
                idx_median = (outputs.shape[1] - 3) // 2 + 1
                MAE = (torch.abs(valid_intensity_quantile_outputs[:,idx_median,:].flatten() - valid_intensity_targets))[mask2].mean()


            quantile_loss = self.quantile_loss(valid_intensity_quantile_outputs, valid_intensity_targets.view(bs,1,-1))
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * quantile_loss
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)
            
            if display_other_stats:
                with torch.no_grad():
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2

                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()

                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()                    
                return MAE.detach(), segmentation_loss.detach(), loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:              
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()
            

        elif self.mode == 'CE_maskedMSE_maskedQ':
            valid_intensity_MSE_outputs = outputs[:, 2, :][mask.squeeze(1)]
            mask3 = valid_intensity_targets > 0
            MSE =  ((valid_intensity_MSE_outputs - valid_intensity_targets)**2)[mask3].mean() 
            
            valid_intensity_quantile_outputs = []
            for i in range(3, outputs.shape[1]):
                valid_intensity_quantile_outputs.append(outputs[:, i, :][mask.squeeze(1)][mask3].unsqueeze(dim=0))
            valid_intensity_quantile_outputs = torch.cat(valid_intensity_quantile_outputs, dim=0)
            quantile_loss = self.quantile_loss(valid_intensity_quantile_outputs.unsqueeze(0), valid_intensity_targets[mask3].unsqueeze(0).unsqueeze(0))
            loss = 1/(2*p[0]**2) * segmentation_loss + 1/(2*p[1]**2) * MSE + 1/(2*p[2]**2) * quantile_loss
            loss += torch.log(1 + p[0]**2 + p[1]**2 + p[2]**2)


            if display_other_stats:
                with torch.no_grad():
                    idx_median = (outputs.shape[1] - 3) // 2 + 1
                    valid_intensity_MAE_outputs = outputs[:, 2 + idx_median, :][mask.squeeze(1)]                
                    MAE = (torch.abs(valid_intensity_MAE_outputs.detach() - valid_intensity_targets.detach()))
                    MSE_tot = MSE.detach()
                    MSE = (valid_intensity_MSE_outputs.detach() - valid_intensity_targets.detach())**2

                    # calc MSE preds
                    preds0 = valid_segmentation_outputs.argmax(dim=1).flatten() == 0
                    MSE_preds = copy.deepcopy(MSE)
                    MSE_preds[preds0] = 0
                    MSE_preds = MSE_preds[mask2].mean()
                    MAE[preds0] = 0
                    MAE = MAE[mask2].mean()
                    # Calc MSE gt
                    mask_gt = valid_intensity_targets == 0
                    MSE_gt = copy.deepcopy(MSE)
                    MSE_gt[mask_gt] = 0
                    MSE_gt = MSE_gt[mask2].mean()                    
                return MAE, segmentation_loss.detach(), loss, cm, MSE_preds, MSE_preds, MSE_gt, MSE_tot 
                
            else:   
                with torch.no_grad():
                    idx_median = (outputs.shape[1] - 3) // 2 + 1
                    valid_intensity_MAE_outputs = outputs[:, 2 + idx_median, :][mask.squeeze(1)]                
                    MAE = (torch.abs(valid_intensity_MAE_outputs - valid_intensity_targets))
                    MSE = (valid_intensity_MSE_outputs - valid_intensity_targets)**2
                    preds = valid_segmentation_outputs.argmax(dim=1).flatten()
                    # print(preds.shape, MAE.shape)
                    MAE[preds == 0] = 0
                    MSE[preds == 0] = 0 
                    MAE = MAE[mask2].mean()
                    MSE = MSE[mask2].mean()            
                return MAE.detach(),  segmentation_loss.detach(), loss, cm, MSE.detach()
                                                        


def compute_metrics(confusion_matrix):
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_ratio = fp / (tp + fp) if (tp + fp) > 0 else 0

    return accuracy, csi, sensitivity, specificity, false_alarm_ratio










            
