# -*- coding: utf-8 -*-
"""Utils of loading config file."""
# -*- coding: utf-8 -*-
import yaml


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
