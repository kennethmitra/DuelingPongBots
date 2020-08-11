import os
import torch


class MemoryGrabber():
    def __init__(self):
        self.mem = os.popen(
            '"<path\to\NVSMI>\nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(
            ",")

    def clear_mem(self,percent_mem):
        total, used = self.mem

        total = int(total)
        used = int(used)

        max_mem = int(total * percent_mem)
        block_mem = max_mem - used

        x = torch.rand((256, 1024, block_mem)).cuda()
        del x
