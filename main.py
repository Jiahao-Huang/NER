import os
import torch
import logging
import hydra
import torch.nn as nn
from torch import optim
from hydra import utils
from torch.utils.data import DataLoader

from preprocess import preprocess
from dataset import NERDataset

logger = logging.getLogger(__name__)

@hydra.main(config_path='config/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda", cfg.gpu_id)
    else:
        device = "cpu"
    logger.info(f"device: {device}")
    
    if cfg.preprocess:
        corpus = preprocess(cfg)

    train_pkl_file = open(os.path.join(cfg.cwd, f"data/out/{cfg.dataset}/train.pkl"), "rb")
    test_pkl_file = open(os.path.join(cfg.cwd, f"data/out/{cfg.dataset}/test.pkl"), "rb")

    train_dataset = NERDataset(train_pkl_file)
    test_dataset = NERDataset(test_pkl_file)
    
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, )
    test_dataloader = DataLoader(test_dataset, cfg.batch_size, shuffle=False, )

if __name__ == "__main__":
    main()

