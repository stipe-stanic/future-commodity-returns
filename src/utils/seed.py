import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PTYONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return seed
