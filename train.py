from trainer.trainer import randomLayerWeightsTrainer
from utils.parser import get_parser
import os
import torch
import numpy as np
import random

def seed_torch(seed=2024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     

def main():
    seed_torch()
    parser = get_parser()
    config = parser.parse_args()
    trainer = randomLayerWeightsTrainer(config)
    
    trainer.train()
    
if __name__ == "__main__":
    main()