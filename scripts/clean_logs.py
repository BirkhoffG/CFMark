import os
import shutil


logs_dir = 'logs'
for root, dirs, files in os.walk(logs_dir):
    if len(files) == 1 and files[0] == 'hparams.yaml':
        shutil.rmtree(root)
            
