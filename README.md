# Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts

Kiyoon Kim, Shreyank N Gowda, Panagiotis Eustratiadis, Antreas Antoniou, Robert B Fisher  
In ICPRAI 2024. [`arXiv`](https://arxiv.org/abs/2401.11406) [`springer`](https://link.springer.com/chapter/10.1007/978-981-97-8702-9_13)

<img src="https://github.com/user-attachments/assets/618213c5-2579-4ee1-bf26-3aa2d219ed56" alt="Adversarial Augmentation" width="800"/>

## Dataset downloads (matching classes only)

- [HMDB-Kinetics-28.csv](https://github.com/kiyoon/video-adversarial-augmentation/files/15058451/hmdb-kinetics-28.csv)

- [UCF-Kinetics-65.csv](https://github.com/kiyoon/video-adversarial-augmentation/files/15058460/ucf-kinetics-65.csv)


## Running the code

```bash
pip install -e .
cp tools/run.env tools/.env
vi tools/.env  # change the settings here

# You can change the hyperparameters using environment variables like below
export ML_num_epochs=200
export ML_batch_size=16
export ML_lr_model=1e-4
# ... see more in BaseConfig in tools/run_aug.py

accelerate launch tools/run_aug.py
```

NOTE: you don't need to download the datasets as they will be downloaded automatically.

## Citing the paper

If you find our work or code useful, please cite:

```BibTeX
@InProceedings{kim2024videoadversarial,
author="Kim, Kiyoon
and Gowda, Shreyank N.
and Eustratiadis, Panagiotis
and Antoniou, Antreas
and Fisher, Robert B.",
editor="Wallraven, Christian
and Liu, Cheng-Lin
and Ross, Arun",
title="Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts",
booktitle="Pattern Recognition and Artificial Intelligence",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="186--200",
isbn="978-981-97-8702-9"
}
```
