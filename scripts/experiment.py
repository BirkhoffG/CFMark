from privacy.pipeline import *
from privacy.watermark import WatermarkConfig
from privacy.attack import extract
from privacy.logger import Logger, WandbLogger, PrintLogger
from privacy.data_module import get_datamodule_and_ml_module
from privacy.detect import DetectionConfig
from relax.import_essentials import *
from relax.data_utils import SoftmaxTransformation, GumbelSoftmaxTransformation
from relax.ml_model import MLModule
import relax
from relax.methods import *
import pydantic
import pydantic_argparse
import pprint
import yaml
import haiku as hk
from devtools import pprint


# https://github.com/huggingface/datasets/blob/main/benchmarks/benchmark_iterating.py
CONFIGS_BASEPATH, CONFIGS_FILENAME = os.path.split(__file__)
CONFIGS_FILE_PATH = os.path.join(
    CONFIGS_BASEPATH, "configs", CONFIGS_FILENAME.replace(".py", ".yaml")
)


class ExperimentConfig(pydantic.BaseModel):

    data: str = "adult"
    cf_method_cls: str = "DiverseCF"
    cf_args: Dict[str, Any] = {"lambda_1": 10.0}
    
    # 1. watermarking CFs
    watermark_conifg: Dict[str, Any] = WatermarkConfig(
        batch_size=64, steps=50, eps=0.05, k=10, n_ensembels=4, lr=0.03, perturb_categorical=False, lambdas=(2., 1., 1.)
    ).dict()
    cat_constraints_fn: Literal['softmax', 'gumbel_softmax'] = 'softmax'

    # 2. attack method
    attack_method_cls: Literal[
        'NaturalTraining', 'QueryModelExtraction', 'ModelExtractionCF', 'DualCFAttack'] = "ModelExtractionCF"
    attack_ml_args: Dict[str, Any] = {
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"], "lr": 0.01,
    }
    n_queries: int = 128
    use_default_exps: bool = False # Use default generated w_cfs for attack
    use_watermarked: bool = True
    # 3. detect watermark
    detect_config: Dict[str, Any] = DetectionConfig(frac=0.1, tau=0.15).dict()
    # Configs
    cfg_name = CONFIGS_FILENAME.replace(".py", ".yaml")
    logger_type: Literal['wandb', 'print', 'local'] = "local"
    save_cfg: bool = False
    global_seed: int = 42

    @property
    def global_rng(self) -> jrand.PRNGKey:
        return jrand.PRNGKey(self.global_seed)

    @property
    def cf_module(self):
        cls = getattr(relax.methods, self.cf_method_cls)
        return cls(self.cf_args)
    
    @property
    def logger(self) -> Logger:
        if self.logger_type == 'wandb':
            return WandbLogger(
                project='privacy', user_name='BirkhoffG',
                experiment_name=self.cfg_name.replace('.yaml', ''), 
                hparams=self.dict(),
            )
        elif self.logger_type == 'print':
            return PrintLogger(self.cfg_name.replace('.yaml', ''), hparams=self.dict())
        return Logger(self.cfg_name.replace('.yaml', ''), hparams=self.dict())

    @property
    def attack_module(self):
        cls = getattr(extract, self.attack_method_cls)
        return cls(
            attack_model=MLModule(self.attack_ml_args),
        )
    
    @property
    def cat_transformation(self):
        """Use softmax or gumbel softmax for applying constraints to categorical data."""

        if self.cat_constraints_fn == 'softmax':
            return "ohe"
        elif self.cat_constraints_fn == 'gumbel_softmax':
            return "gumbel"

    def to_yaml(self):
        configs_file_path = os.path.join(CONFIGS_BASEPATH, "configs", self.cfg_name)
        with open(configs_file_path, "w") as f: 
            yaml.dump(self.dict(), f)

    @classmethod
    def from_yaml(cls, obj=None):
        """Load default configs from yaml and override configs with obj."""
        if obj is not None:
            config_filename = obj.get("cfg_name", 
                                      CONFIGS_FILENAME.replace(".py", ".yaml"))
        path = os.path.join(CONFIGS_BASEPATH, "configs", config_filename)
        
        # handle case where file does not exist
        if not os.path.exists(path):
            return cls.parse_obj(obj)
        
        # load from yaml and update with obj
        with open(path, "r") as f:
            yaml_args = yaml.safe_load(f)
            if obj is not None:
                yaml_args.update(obj)
            return cls.parse_obj(yaml_args)


def set_transformation(dm: relax.DataModule, transformation):
    # Cannot set OneHotTransformation to the data module for some reasons
    if isinstance(transformation, SoftmaxTransformation): return dm

    dm = dm.set_transformations({
        feat.name: transformation
        for feat in dm.features if feat.is_categorical
    })
    return dm


def main(config: ExperimentConfig):
    pprint(config)
    # Load data
    dm, ml_module = get_datamodule_and_ml_module(config.data)
    dm = set_transformation(dm, config.cat_transformation)
    print(f"shape: {dm.xs.shape}")
    cf_module = config.cf_module
    attack_fn = config.attack_module
    if config.use_default_exps:
        output = attack_default_exps(
            default_dir=Path(f"logs/{config.logger.name}/default"),
            dm=dm, cf_module=cf_module, attack_fn=attack_fn,
            n_queries=config.n_queries, ml_module=ml_module,
            logger=config.logger,
            train_kwargs={'epochs': 100, 'validation_split': 0., 'batch_size': 32, 'verbose': False},
            detect_config=config.detect_config,
            use_watermark_to_extract=config.use_watermarked,
            rng_key=config.global_rng,
        )
    else:
        output = pipeline(
            dm=dm, 
            cf_module=cf_module, 
            attack_fn=attack_fn, 
            ml_module=ml_module, 
            logger=config.logger,
            n_queries=config.n_queries, 
            train_kwargs={'epochs': 100, 'validation_split': 0., 'batch_size': 32, 'verbose': False},
            watermark_kargs_or_config=config.watermark_conifg,
            detect_config=config.detect_config,
            use_watermark_to_extract=config.use_watermarked,
            rng_key=config.global_rng,
        )
    
    # Save configs
    if config.save_cfg:
        config.to_yaml()
    return output


if __name__ == "__main__":
    # Parse command line arguments
    parser = pydantic_argparse.ArgumentParser(
        ExperimentConfig,
        description="Run experiment.",
    )
    # Convert Argparse Namespace to Dictionary
    args = pydantic_argparse.utils.namespaces.to_dict(parser.parse_args())
    # If the config file exists, load it as a Pydantic model instance,
    # and update it with the command line arguments.
    if Path(CONFIGS_FILE_PATH).exists():
        config = ExperimentConfig.from_yaml(args)
        print(f"Loaded configs from {CONFIGS_FILE_PATH}")
    else:
        config = ExperimentConfig(**args)

    main(config)
