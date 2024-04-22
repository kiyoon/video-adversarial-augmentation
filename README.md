# Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts

Kiyoon Kim, Shreyank N Gowda, Panagiotis Eustratiadis, Antreas Antoniou, Robert B Fisher  
In ICPRAI 2024. [`arXiv`](https://arxiv.org/abs/2401.11406)

## Dataset downloads (labels only)
- [HMDB-Kinetics-28]()
- [UCF-Kinetics-65]()

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


## Citing the paper

If you find our work or code useful, please cite:

```BibTeX

```
