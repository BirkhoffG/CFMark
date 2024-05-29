import os
from .experiment import ExperimentConfig, main
import jax.random as jrand
from relax.utils import save_pytree
from pathlib import Path

N_EXPERIMENTS = 10


def get_path_relative_to_configs(absolute_path: str):
    """Returns the path relative to the 'scripts/configs' directory."""

    scripts_configs_dir = "scripts/configs"
    # Check if the absolute path starts with the scripts/configs directory
    if absolute_path.startswith(scripts_configs_dir):
        relative_path = absolute_path[len(scripts_configs_dir) + 1:]  # +1 to remove the leading '/'
        return relative_path
    else:
        return None  # The path is not within 'scripts/configs'


def run(file: str):
    config_file_path = os.path.join(root, file)
    cfg_name = get_path_relative_to_configs(config_file_path)
    print(f"Running {cfg_name}...")
    method_name, data_name = cfg_name.replace('.yaml', '').split('/')
    # Skip methods and datasets that are not in the constraints
    if 'method_name' in constraints.keys() and method_name not in constraints['method_name']:
        print(f"Skipping {cfg_name}...")
        return
    if 'data' in constraints.keys() and data_name not in constraints['data']:                   
        print(f"Skipping {cfg_name}...")
        return
    print(f"Running {cfg_name}...")

    for attack_method in constraints['attack_methods']:
        seeds = jrand.randint(jrand.PRNGKey(1234), (N_EXPERIMENTS,), 0, 2**16 - 1)
        for seed in seeds:
            config = ExperimentConfig.from_yaml({"cfg_name": cfg_name,
                                        #  'NaturalTraining', 'QueryModelExtraction', 'ModelExtractionCF', 'DualCFAttack'
                                         "attack_method_cls": attack_method,
                                         "use_default_exps": True, 
                                         'logger_type': 'print',
                                         'global_seed': seed,
                                         "save_cfg": False})
            output = main(config)
            results.append({
                'attack': attack_method,
                'cfg_name': cfg_name,
                'seed': seed,
                'results': output.detect_results
            })
    

# constraints specify which data or methods to run
# If a constraint is not specified, all methods or datasets will be run
constraints = {
    # 'data': ['dummy', 'heloc', 'credit'],
    'data': ['cancer'],
    # 'data': ['dummy'],
    # 'method_name': ['GrowingSphere', 'CCHVAE'] #, 'GrowingSphere'],
    'attack_methods': ['QueryModelExtraction', 'ModelExtractionCF', 'DualCFAttack']
}
# store results
results = []

# Find configuration files
config_dir = "scripts/configs"  # Path relative to run_all.py
for root, dirs, files in os.walk(config_dir):
    for file in files:
        try:
            if file.endswith(".yaml") and 'rand' not in root:
                run(file)
        except Exception as e:
            print(f"Error in {file}: {e}")

# mk dir
path = Path(f'logs/meta_test-{N_EXPERIMENTS}-{constraints["data"][0]}')
path.mkdir(parents=True, exist_ok=True)
save_pytree(results, path)
