import argparse
import re
import yaml
import os
from pathlib import Path

# Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
# (more info: https://github.com/pytorch/pytorch/issues/1355)
import cv2
cv2.setNumThreads(0)

from src.registry import TASKS
from src.constructor.config_structure import TrainConfigParams
from src.constructor import create_trainer


def load_config(path):
    # environment variables substitution, so ${ENV_VAR} is substituted in every scalar value
    env_matcher = re.compile(r'\$\{([^}^{]+)\}')
    
    def _env_constructor(loader, node, deep=False):
        value = loader.construct_scalar(node)
            
        match = env_matcher.search(value)
        if match:
            env_var = match.group()[2:-1]
            full_var = value[:match.start()] + os.environ.get(env_var, '') + value[match.end():]

            return full_var

        return value
    
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, _env_constructor, yaml.SafeLoader)
    
    path = Path(path)
    if not path.exists() or path.suffix != ".yml":
        raise Exception('You must provide path to existing .yml file')

    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="Path to .yml file with configuration parameters.")
    parser.add_argument('--job_link', type=str,
                        help="sagemaker job name, if running localy set to 'local'", default='local')
    args = parser.parse_args()
    config_path = args.config

    config_yaml = load_config(config_path)
    config = TrainConfigParams(**config_yaml)

    model = TASKS.get(config.task.name)(config)
    trainer = create_trainer(config, str(args.job_link))
    trainer.fit(model)
