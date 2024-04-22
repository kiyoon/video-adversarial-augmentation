from hydra_zen import store
from torch.optim import SGD, AdamW

optimizer_store = store(group="optimizer")
optimizer_store(AdamW, lr=1e-5, weight_decay=1e-5)
optimizer_store(SGD, lr=1e-5, momentum=0.9, weight_decay=5e-4)
