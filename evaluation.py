import os
import csv
import time
import torch
import clip
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from ptflops import get_model_complexity_info
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
from utils.Text_Prompt import *
from datasets import Action_DATASETS
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime


    
def print_model_param_stats(model, name="CLIP"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Total Params     : {total / 1e6:.2f} M")
    print(f"[{name}] Trainable Params : {trainable / 1e6:.2f} M")


# === UCF101 5-카테고리 매핑 (라벨 파일 표기와 동일) ===
CATEGORY_MAP = {
    "Human-Object Interaction": [
        "ApplyEyeMakeup","ApplyLipstick","BlowDryHair","BrushingTeeth","CuttingInKitchen",
        "Hammering","HulaHoop","JugglingBalls","JumpRope","Knitting","Mixing","MoppingFloor",
        "Nunchucks","PizzaTossing","ShavingBeard","SkateBoarding","SoccerJuggling","Typing",
        "WritingOnBoard","YoYo"
    ],
    "Body-Motion Only": [
        "BabyCrawling","BlowingCandles","BodyWeightSquats","HandStandPushups","HandstandWalking",
        "JumpingJack","Lunges","PullUps","PushUps","RockClimbingIndoor","RopeClimbing",
        "Swing","TaiChi","TrampolineJumping","WalkingWithDog","WallPushups"
    ],
    "Human-Human Interaction": [
        "BandMarching","Haircut","HeadMassage","MilitaryParade","SalsaSpin"
    ],
    "Playing Musical Instruments": [
        "Drumming","PlayingCello","PlayingDaf","PlayingDhol","PlayingFlute",
        "PlayingGuitar","PlayingPiano","PlayingSitar","PlayingTabla","PlayingViolin"
    ],
    "Sports": [
        "Archery","BalanceBeam","BaseballPitch","Basketball","BasketballDunk","BenchPress",
        "Biking","Billiards","Bowling","BoxingPunchingBag","BoxingSpeedBag","BreastStroke",
        "CleanAndJerk","CliffDiving","CricketBowling","CricketShot","Diving","Fencing",
        "FieldHockeyPenalty","FloorGymnastics","FrisbeeCatch","FrontCrawl","GolfSwing",
        "HammerThrow","HighJump","HorseRace","HorseRiding","IceDancing","JavelinThrow",
        "Kayaking","LongJump","ParallelBars","PoleVault","PommelHorse","Punch","Rafting",
        "Rowing","Shotput","Skiing","Skijet","SkyDiving","SoccerPenalty","StillRings",
        "SumoWrestling","Surfing","TableTennisShot","TennisSwing","ThrowDiscus",
        "UnevenBars","VolleyballSpiking"
    ]
}

CAT_ORDER = list(CATEGORY_MAP.keys())

ALIASES_UCF = {
    "HandstandPushups": "HandStandPushups",
    "WalkingWithADog":  "WalkingWithDog",
    "JetSki":           "Skijet",
    "Billiard":         "Billiards",
    "Nunchuks":         "Nunchucks",
    "Nunchuck":         "Nunchucks",
    "MixingBatter":     "Mixing",
}
def _canon(name: str) -> str:
    name = name.strip().replace(" ", "").replace("_", "")
    for a,b in ALIASES_UCF.items():
        if name.lower() == a.replace(" ","").replace("_","").lower():
            return b
    return next((lbl for lbl in CATEGORY_MAP["Human-Object Interaction"]
                 + CATEGORY_MAP["Body-Motion Only"]
                 + CATEGORY_MAP["Human-Human Interaction"]
                 + CATEGORY_MAP["Playing Musical Instruments"]
                 + CATEGORY_MAP["Sports"]
                 if lbl.replace(" ","").replace("_","").lower()==name.lower()), 
                name)

def _name2cat():
    m = {}
    for cat, names in CATEGORY_MAP.items():
        for n in names:
            m[_canon(n)] = cat
    return m

def _ensure_names(class_names):
    return [_canon(x) for x in class_names]

def confusion_matrix_from_ids(gt_ids, pred_ids, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt_ids, pred_ids):
        cm[int(g), int(p)] += 1
    return cm

def aggregate_by_category(cm_51, class_names):
    cname = _ensure_names(class_names)
    name2cat = _name2cat()
    cat_idx = {c: i for i, c in enumerate(CAT_ORDER)}
    cm5 = np.zeros((5, 5), dtype=np.int64)
    for i, gi_name in enumerate(cname):
        gi_cat = name2cat.get(gi_name)
        if gi_cat is None: 
            continue
        gi = cat_idx[gi_cat]
        for j, pj_name in enumerate(cname):
            pj_cat = name2cat.get(pj_name)
            if pj_cat is None:
                continue
            pj = cat_idx[pj_cat]
            cm5[gi, pj] += cm_51[i, j]
    return cm5

def row_normalize(cm):
    import numpy as np
    cm = cm.astype(float)
    rowsum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.divide(cm, rowsum, where=rowsum!=0)
    return out

def top_confusions_from_cm(cm_51, class_names, k=5):
    import numpy as np, pandas as pd
    cname = _ensure_names(class_names)
    data = []
    totals = cm_51.sum(axis=1, keepdims=True)
    for i in range(cm_51.shape[0]):
        row = cm_51[i].copy()
        row[i] = 0
        if row.sum() == 0:
            continue
        top_idx = np.argsort(row)[::-1][:k]
        for j in top_idx:
            cnt = int(row[j])
            if cnt == 0:
                continue
            rate = cnt / totals[i][0] if totals[i][0] > 0 else 0.0
            data.append([cname[i], cname[j], cnt, rate])
    df = pd.DataFrame(data, columns=["src_class", "dst_class", "count", "rate"])
    return df.sort_values(["count", "rate"], ascending=[False, False])



class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


inv_transform = T.Compose([
    T.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
    T.Normalize(mean=[-0.5]*3, std=[1.]*3),
])

class ActionCLIPFull(nn.Module):
    def __init__(self, model_image, fusion_model):
        super().__init__()
        self.model_image = model_image
        self.fusion_model = fusion_model

    def forward(self, image):  # (B, T, C, H, W)
        b, t, c, h, w = image.size()
        image = image.view(b * t, c, h, w)
        feats = self.model_image(image)
        feats = feats.view(b, t, -1)
        return self.fusion_model(feats)

def visualize_and_save_wrong_cases(wrong_cases, class_names, save_dir="./wrong_cases", num_samples=20):
    fail_counter = Counter([wc['true_label'] for wc in wrong_cases])
    class_order = [cid for cid, _ in fail_counter.most_common()]

    wrong_cases_sorted = sorted(
        wrong_cases,
        key=lambda wc: class_order.index(wc['true_label'])
    )
    selected = wrong_cases_sorted[:min(len(wrong_cases_sorted), num_samples)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "wrong_cases.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Index", "True Label", "Predicted Label",
            "Top1 Confidence (%)", "Top-5 Predictions (label:prob%)"
        ])

        for i, sample in enumerate(selected):
            img = inv_transform(sample['image']).permute(1,2,0).numpy().clip(0,1)
            true_cls = class_names[sample['true_label']]
            pred_cls = class_names[sample['pred_label']]
            conf = sample['confidence'] * 100
            fname = f"{i:02d}_true_{true_cls}_pred_{pred_cls}_{conf:.1f}.png"
            fpath = os.path.join(out_dir, fname)

            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.title(f"T:{true_cls} | P:{pred_cls} ({conf:.1f}%)", fontsize=10)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fpath)
            plt.close()

            top5_ids   = sample.get("top5_ids", [])
            top5_probs = sample.get("top5_probs", [])
            top5_str   = "; ".join(
                f"{class_names[c]}:{p*100:.1f}%"
                for c,p in zip(top5_ids, top5_probs)
            )

            writer.writerow([
                i, true_cls, pred_cls,
                f"{conf:.1f}", top5_str
            ])

    print(f"Save {len(selected)} failed images → '{out_dir}/'")
    print(f"Save failure log as CSV → '{csv_path}'")


def convert_module_to_half(module):
    for child in module.children():
        convert_module_to_half(child)
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
        module.half()


def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, class_names):
    model.eval()
    fusion_model.eval()
    num, corr_1, corr_5 = 0, 0, 0
    video_stats = {
        cid: {"total": 0, "correct1": 0, "correct5": 0}
        for cid in range(len(class_names))
    }

    wrong_cases = []
    all_gt, all_pred = [], []


    total_time = 0
    num_batches = 0
    peak_memory = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)

        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            #print(f"Batch {iii} loaded")
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            image_raw = image.clone()
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            #similarity = (100.0 * image_features @ text_features.T)
            #similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            #similarity = similarity.mean(dim=1)

            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1)
            similarity = similarity.mean(dim=1)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)

            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            total_time += batch_time
            num_batches += 1
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**2)

            num += b
            for i in range(b):
                true_id = int(class_id[i].item())
                pred_id = int(indices_1[i].item())
                video_stats[true_id]["total"]   += 1
                if pred_id == true_id:
                    video_stats[true_id]["correct1"] += 1
                if true_id in indices_5[i]:
                    video_stats[true_id]["correct5"] += 1
                prob = float(similarity[i][pred_id].item())
                if pred_id == true_id:
                    corr_1 += 1
                else:
                    top5_ids = [int(idx.item()) for idx in indices_5[i]]
                    top5_probs = [float(similarity[i][idx].item()) for idx in top5_ids]
                    wrong_cases.append({
                        'image': image_raw[i][:3].cpu(),
                        'true_label': true_id,
                        'pred_label': pred_id,
                        'confidence': prob,
                        'top5_ids': top5_ids,
                        'top5_probs': top5_probs
                    })
                if true_id in indices_5[i]:
                    corr_5 += 1
                    
                all_gt.append(true_id)
                all_pred.append(pred_id)
                
        #expected = len(val_loader.dataset) if not drop_last_flag \
        #else (len(val_loader.dataset) // config.data.batch_size) * config.data.batch_size
        #print(f"[CHECK] evaluated={expected} saved={len(all_gt)}")
        #assert len(all_gt) == expected, "Saved rows != evaluated samples"


    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    latency = (total_time / num_batches) / config.data.batch_size * 1000

    print(f"[Top1] {top1:.2f} | [Top5] {top5:.2f} | [Latency] {latency:.2f} ms/sample | [VRAM] {peak_memory:.2f} MB")
    wandb.log({"top1": top1, "top5": top5, "latency_ms": latency, "peak_vram_MB": peak_memory})

    visualize_and_save_wrong_cases(wrong_cases, class_names)

    df_pred = pd.DataFrame({"gt_id": all_gt, "pred_id": all_pred})
    out_dir = "./analysis_outputs"
    os.makedirs(out_dir, exist_ok=True)
    pred_csv = os.path.join(out_dir, "ucf_preds_top1.csv")
    df_pred.to_csv(pred_csv, index=False)
    print(f"[SAVE] per-sample predictions → {pred_csv}")

    cm_51 = confusion_matrix_from_ids(np.array(all_gt, dtype=int), np.array(all_pred, dtype=int), n_classes=len(class_names))
    pd.DataFrame(cm_51, index=[_canon(x) for x in class_names], columns=[_canon(x) for x in class_names])\
        .to_csv(os.path.join(out_dir, "cm_51x51.csv"))
    print(f"[SAVE] cm_51x51.csv")

    cm_5 = aggregate_by_category(cm_51, class_names)
    pd.DataFrame(cm_5, index=CAT_ORDER, columns=CAT_ORDER)\
        .to_csv(os.path.join(out_dir, "cm_5x5.csv"))
    pd.DataFrame(row_normalize(cm_5), index=CAT_ORDER, columns=CAT_ORDER)\
        .to_csv(os.path.join(out_dir, "cm_5x5_row_normalized.csv"))
    print(f"[SAVE] cm_5x5.csv, cm_5x5_row_normalized.csv")

    df_top = top_confusions_from_cm(cm_51, class_names, k=5)
    df_top.to_csv(os.path.join(out_dir, "top_confusions_per_class.csv"), index=False)
    print(f"[SAVE] top_confusions_per_class.csv")

    try:
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(row_normalize(cm_5), interpolation='nearest')  # 색상 지정 안 함(기본)
        ax.set_xticks(range(len(CAT_ORDER))); ax.set_yticks(range(len(CAT_ORDER)))
        ax.set_xticklabels(CAT_ORDER, rotation=45, ha="right"); ax.set_yticklabels(CAT_ORDER)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        for i in range(cm_5.shape[0]):
            for j in range(cm_5.shape[1]):
                val = row_normalize(cm_5)[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "cm_5x5.png"), dpi=200)
        plt.close(fig)
        print(f"[SAVE] cm_5x5.png")
    except Exception as e:
        print(f"[WARN] 5x5 plot failed: {e}")

    out_dir = "./classwise_video_performance"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "classwise_video_accuracy.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ClassID", "ClassName", "NumVideos", "Top1(%)", "Top5(%)"])
        for cid, stats in video_stats.items():
            total = stats["total"]
            if total == 0:
                acc1 = acc5 = 0.0
            else:
                acc1 = stats["correct1"] / total * 100
                acc5 = stats["correct5"] / total * 100
            writer.writerow([cid, class_names[cid], total, f"{acc1:.2f}", f"{acc5:.2f}"])
    print(f"Save per-class video performance → '{csv_path}'")
    return top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = DotMap(yaml.safe_load(f))

    working_dir = os.path.join('./exp', config.network.type, config.network.arch, config.data.dataset, args.log_time)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    wandb.init(project=config.network.type, name='{}_{}_{}_{}'.format(args.log_time, config.network.type, config.network.arch, config.data.dataset))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(
        config.network.arch, device=device, jit=False, tsm=config.network.tsm,
        T=config.data.num_segments, dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
        joint=config.network.joint,
        use_text_adapter=config.solver.use_text_adapter,
        text_adapter_layers=config.solver.text_adapter_layers,
        text_adapter_dim=config.solver.text_adapter_dim,
        use_visual_adapter=config.solver.use_visual_adapter,
        visual_adapter_layers=config.solver.visual_adapter_layers,
        visual_adapter_dim=config.solver.visual_adapter_dim
    )
    
    print_model_param_stats(model, name="CLIP")

    model = model.half()
    convert_module_to_half(model)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.float()

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    print_model_param_stats(fusion_model, name="Fusion")
    fusion_model = fusion_model.half()
    convert_module_to_half(fusion_model)
    for m in fusion_model.modules():
        if isinstance(m, nn.LayerNorm):
            m.float()

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()


    def get_flops_params(model):
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model, (8, 3, 224, 224),  # 8 segments 기준
                as_strings=True,
                print_per_layer_stat=False, verbose=False)
        return macs, params

    flop_model = ActionCLIPFull(model_image.module, fusion_model.module)
    macs, params = get_flops_params(flop_model)
    print(f"[FLOPs]     {macs}")
    print(f"[Params]    {params}")

    if config.solver.resume:
        ckpt = torch.load(config.solver.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        fusion_model.load_state_dict(ckpt['fusion_model_state_dict'], strict=False)
        del ckpt

    transform_val = get_augmentation(False, config)
    val_data = Action_DATASETS(
        list_file=config.data.val_list, labels_file=config.data.label_list,
        root=config.data.root, num_segments=config.data.num_segments,
        new_length=config.data.seg_length, image_tmpl=config.data.image_tmpl,
        transform=transform_val, random_shift=False, test_mode=True,
        index_bias=config.data.index_bias)
        
    print("VAL LIST PATH:", config.data.val_list)
    print("LABEL LIST PATH:", config.data.label_list)
    print("→ Loaded class count:", len(pd.read_csv(config.data.label_list)))
    print("→ Loaded sample count:", len(val_data))

    val_loader = DataLoader(val_data, batch_size=config.data.batch_size,
                            num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    classes, num_text_aug, text_dict = text_prompt(val_data)
    label_df = pd.read_csv(config.data.label_list)
    class_names = label_df.sort_values(label_df.columns[0]).iloc[:, 1].tolist()

    validate(config.solver.start_epoch, val_loader, classes, device,
             model, fusion_model, config, num_text_aug, class_names)


if __name__ == '__main__':
    main()