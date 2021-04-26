import os
import torch
import logging
import hydra
import torch.nn as nn
from torch import optim
from hydra import utils

from preprocess import preprocess

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
    
    corpus = preprocess(cfg)
    

if __name__ == "__main__":
    main()

