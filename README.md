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

Results (Summary)
In-domain Performance (Top-1 Accuracy, %)
Dataset	Shot	Base	EA-ActionCLIP	Δ
HMDB51	1	55.1	53.5	-1.6
	3	63.1	61.8	-1.3
	5	65.5	62.4	-3.1
	7	67.7	66.5	-1.1
UCF101	1	86.8	84.2	-2.6
	3	93.3	91.6	-1.7
	5	93.4	93.3	-0.1
	7	94.9	93.9	-1.0
In-domain Efficiency
Metric	Base	EA-ActionCLIP	Reduction
GPU Memory (HMDB51)	avg 51%	avg 11%	78.4% ↓
GPU Memory (UCF101)	avg 41%	avg 19%	53.7% ↓
Process Memory	~5.4 GB	~1.7 GB	68% ↓
Training Time	65–117 min	40–73 min	≈38% ↓
Trainable Parameters	149.6 M	0.79 M	99.5% ↓
Cross-domain Performance (Top-1 Accuracy, %)
Transfer	Summary
HMDB51 → UCF101	EA-ActionCLIP outperformed the base by +8–9% (avg)
UCF101 → HMDB51	EA-ActionCLIP higher at 1-shot (+4.8%), then slight declines (–0.5% to –2.2%) as shots increase

Detailed Top-5 accuracy and per-shot tables are available in the dissertation.
