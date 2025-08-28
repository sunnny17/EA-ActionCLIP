# EA-ActionCLIP

An adapter-augmented variant of ActionCLIP for few-shot action recognition (FSAR).  
This repository contains code, configs, and scripts to reproduce the experiments from my MSc dissertation.

<img width="3875" height="1672" alt="Figure3" src="https://github.com/user-attachments/assets/72344272-97ea-4539-8f04-7f138c7c2adc" />

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
  
-> See `environment.yaml` for full details.

---

## Datasets
- **HMDB51**: [link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)  
- **UCF101**: [link](https://www.crcv.ucf.edu/data/UCF101.php)  

---

## Usage

### Training (example with Slurm on HMDB51)
```bash
sbatch run_hmdb.sh
```bash

### Evaluation (example with UCF101, interactive mode)
python evaluation.py --config configs/ucf101/ucf_test.yaml \
  2>&1 | tee /scratch/NEW_adapter/logs/interactive_$(date +%Y%m%d_%H%M%S).log

---

## Results (Summary)

### In-domain Performance (Top-1 Accuracy, %)

