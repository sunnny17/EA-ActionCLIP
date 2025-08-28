# =============================================================================
# Based on ActionCLIP:
#   "ActionCLIP: A New Paradigm for Action Recognition"
#   Mengmeng Wang, Jiazheng Xing, Yong Liu
#   arXiv:2109.08472
#   https://github.com/sallymmx/ActionCLIP (MIT License)
#
# This repository contains substantial modifications and extensions
# made by Yoonseon Oh (2025), including changes to model architecture,
# training/evaluation pipeline, and efficiency logging.
#
# License: MIT (see LICENSE file in the repository root)
# =============================================================================

import torch

def epoch_saving(epoch, model, fusion_model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'fusion_model_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename