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

import os
def get_model_path(ckpt):
    if os.path.isfile(ckpt):
        return ckpt
    else:
        print('not found pretrained model in {}'.format(ckpt))
        raise FileNotFoundError
