# EA-ActionCLIP

An adapter-augmented variant of ActionCLIP for few-shot action recognition (FSAR).  
This repository contains code, configs, and scripts to reproduce the experiments from my MSc dissertation.

---

## Requirements
- PyTorch >= 1.8
- wandb
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv
  
- See `environment.yaml` for full details.

---

## Datasets
- **HMDB51**: [link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)  
- **UCF101**: [link](https://www.crcv.ucf.edu/data/UCF101.php)  

Expected dataset structure:
