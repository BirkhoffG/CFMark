from .experiment import ExperimentConfig
from pathlib import Path
import os


CONFIGS_BASEPATH, CONFIGS_FILENAME = os.path.split(__file__)

cf_modules = [
    ('DiverseCF', {'lambda_1': 10.0}),
    ('GrowingSphere', {}),
    ('CCHVAE', {})
]
data = [
    # 'dummy', 'adult', 
    # 'german', 'titanic', 
    'heloc', 'credit'
]

for cf_name, cf_args in cf_modules:
    for d in data:
        cfg = ExperimentConfig(
            data=d,
            cf_method_cls=cf_name,
            cf_args=cf_args,
            logger_type='local',
            save_cfg=True,
            cfg_name=f"{cf_name}/{d}.yaml"
        )
        configs_file_path = os.path.join(CONFIGS_BASEPATH, f"configs/{cf_name}")
        Path(configs_file_path).mkdir(parents=True, exist_ok=True)
        cfg.to_yaml()
        print(f"Generated {cfg.cfg_name}")