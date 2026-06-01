import random
import numpy as np
import torch

from utils.running_stats import RunningMeanStd
from utils.curriculum import CurriculumScheduler
from utils.logger import MetricLogger


def set_seed(seed: int):
    """Seed all random sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["RunningMeanStd", "CurriculumScheduler", "MetricLogger", "set_seed"]
