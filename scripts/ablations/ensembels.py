"""Ablation study on the n_ensembels term in the watermarking method"""
import os
from ..experiment import ExperimentConfig, main
from pathlib import Path
from privacy.pipeline import *
from privacy.data_module import get_datamodule_and_ml_module
from privacy.logger import Logger
import shutil
import gc
import jax
from beartype import beartype as typecheck


def get_path_relative_to_configs(absolute_path: str):
    """Returns the path relative to the 'scripts/configs' directory."""

    scripts_configs_dir = "scripts/configs"
    # Check if the absolute path starts with the scripts/configs directory
    if absolute_path.startswith(scripts_configs_dir):
        relative_path = absolute_path[len(scripts_configs_dir) + 1:]  # +1 to remove the leading '/'
        return relative_path
    else:
        return None  # The path is not within 'scripts/configs'

@typecheck
def run(file: str, n_ensembles: int):
    config_file_path = os.path.join(root, file)
    cfg_name = get_path_relative_to_configs(config_file_path)
    print(f"Running {cfg_name}...")
    if 'rand' in cfg_name:
        print(f"Skipping {cfg_name}...")
        return

    method_name, data_name = cfg_name.replace('.yaml', '').split('/')
    # Skip methods and datasets that are not in the constraints
    if 'method_name' in constraints.keys() and method_name not in constraints['method_name']:
        print(f"Skipping {cfg_name}...")
        return
    if 'data' in constraints.keys() and data_name not in constraints['data']:                   
        print(f"Skipping {cfg_name}...")
        return
    print(f"Running {cfg_name}...")
    config = ExperimentConfig.from_yaml({"cfg_name": cfg_name,
                                        #  'QueryModelExtraction', 'ModelExtractionCF', 'DualCFAttack'
                                         "attack_method_cls": 'QueryModelExtraction',
                                         "use_default_exps": True,
                                         "save_cfg": False})
    if n_ensembles > config.watermark_conifg['n_ensembels']:
        print(f"n_ensembles={n_ensembles} might be too large; Skipping {cfg_name}...")
        return
    config.watermark_conifg['n_ensembels'] = n_ensembles
    logger = config.logger
    shutil.rmtree(logger.path)
    setattr(logger, 'path', 
            Path(f'logs/ablation/{config.logger.name}/{config.attack_method_cls}/n_ensembels={n_ensembles}'))
    logger.path.mkdir(parents=True, exist_ok=True)
    dm, ml_module = get_datamodule_and_ml_module(config.data)
    print(f"shape: {dm.xs.shape}")
    cf_module = config.cf_module
    attack_fn = config.attack_module
    print(f'logger path: {logger.path}')
    
    pipeline(
        dm=dm, 
        cf_module=cf_module, 
        attack_fn=attack_fn, 
        ml_module=ml_module, 
        logger=logger, # specify the logger
        n_queries=config.n_queries, 
        train_kwargs={'epochs': 100, 'validation_split': 0., 'batch_size': 32, 'verbose': False},
        watermark_kargs_or_config=config.watermark_conifg,
        detect_config=config.detect_config,
        use_watermark_to_extract=True,
        rng_key=config.global_rng,
    )


# constraints specify which data or methods to run
# If a constraint is not specified, all methods or datasets will be run
constraints = {
    'data': ['credit', 'dummy'],
    # 'method_name': ['GrowingSphere'] #['CCHVAE'] #, 'GrowingSphere'],
}

# Find configuration files
config_dir = "scripts/configs"  # Path relative to run_all.py
for root, dirs, files in os.walk(config_dir):
    for file in files:
        if file.endswith(".yaml"):
            for n_ensembles in [1, 2, 4, 8, 16, 32]:
                run(file, n_ensembles)
                jax.clear_caches()
                gc.collect()
                
