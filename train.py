# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    # 안전하게 YAML 파싱
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        # config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    wandb.init(project=f"{config['network']['type']}_ad",name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)
    config.solver.lr = float(config.solver.lr)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(
        config.network.arch,
        device=device,
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
        use_text_adapter=config.solver.use_text_adapter,
        text_adapter_layers=config.solver.text_adapter_layers,
        text_adapter_dim=config.solver.text_adapter_dim,
        use_visual_adapter=config.solver.use_visual_adapter,
        visual_adapter_layers=config.solver.visual_adapter_layers,
        visual_adapter_dim=config.solver.visual_adapter_dim
    )

    for name, param in model.named_parameters():
        param.requires_grad = False

    if config.solver.use_visual_adapter:
        for name, param in model.named_parameters():
            if 'visual' in name and 'adapter' in name:
                param.requires_grad = True

    if config.solver.use_text_adapter:
        for name, param in model.named_parameters():
            if 'transformer.resblocks' in name and 'adapter' in name:
                param.requires_grad = True

    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True

    if hasattr(model, 'text_projection'):
        model.text_projection.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Confirmation] Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    adapter10 = model.visual.transformer.resblocks[10].adapter
    adapter11 = model.visual.transformer.resblocks[11].adapter
    
    adapter10.scale.data.fill_(5e-4)
    adapter11.scale.data.fill_(1e-3)
    
    print("α layer10:", adapter10.scale.item())
    print("α layer11:", adapter11.scale.item())


    #optimizer = _optimizer(config, model, fusion_model)

    adapter_params = []
    head_params    = []          # logit_scale, text_projection
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'adapter' in n:
                adapter_params.append(p)
            else:
                head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': 1e-3, 'weight_decay': 0.02},
        {'params': head_params,   'lr': 5e-4, 'weight_decay': 0.0},
    ])

    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    train_data = Action_DATASETS(list_file = config.data.train_list, labels_file = config.data.label_list, root = config.data.root, num_segments = config.data.num_segments, new_length = config.data.seg_length, image_tmpl = config.data.image_tmpl, transform = transform_train, random_shift = config.data.random_shift, test_mode = False, index_bias = config.data.index_bias)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS(list_file = config.data.val_list, labels_file = config.data.label_list, root = config.data.root, num_segments = config.data.num_segments, new_length = config.data.seg_length, image_tmpl = config.data.image_tmpl, transform = transform_val, random_shift = False, test_mode = True, index_bias = config.data.index_bias)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        model_text.float()
        model_image.float()
        #clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        #clip.model.convert_weights(model_image)
        #model.dtype = torch.float16

    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(train_data)
    
    
    lr_scheduler = _lr_scheduler(config, optimizer)
    
    for i, g in enumerate(optimizer.param_groups):
        print(f"[Init] param_group[{i}] lr = {g['lr']}")

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug)
        return

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
            #if config.solver.type != 'monitor':
                #if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    #lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            b,t,c,h,w = images.size()
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            total_loss = (loss_imgs + loss_texts)/2
            wandb.log({"lr_adapter": optimizer.param_groups[0]['lr'],"lr_heads"  : optimizer.param_groups[1]['lr'], "train_total_loss": total_loss,"train_loss_imgs": loss_imgs,"train_loss_texts": loss_texts,}, commit=True)          # commit=True 가 기본값이지만 명시해 두면 헷갈림 방지
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                #convert_models_to_fp32(model)
                optimizer.step()
                #clip.model.convert_weights(model)

            if config.solver.type != 'monitor' and ((kkk + 1) == 1 or (kkk + 1) % 10 == 0):
                lr_scheduler.step()

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, optimizer)

if __name__ == '__main__':
    main()
