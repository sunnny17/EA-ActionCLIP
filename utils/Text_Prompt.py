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
import clip

def text_prompt(data):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict