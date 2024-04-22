from hydra_zen import store
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import MultiStepLR

scheduler_store = store(group="scheduler")

scheduler_store(CosineLRScheduler, name="cosine-annealing")
scheduler_store(MultiStepLR, name="MultiStepLR", milestones=[5000, 8000], gamma=0.3)
