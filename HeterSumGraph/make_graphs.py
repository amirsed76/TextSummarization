import os
import time

for i in range(88320,287000,1280):
    os.system(f"python train.py --from_instances_index {i}")
