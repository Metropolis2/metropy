import os
import sys

try:
    import tomllib
except ModuleNotFoundError:
    from pip._vendor import tomli as tomllib


def read_config() -> dict:
    """Reads the `config.toml` file from the default path or from the "--config" argument."""
    if len(sys.argv) == 1:
        # Read the config from the default path.
        config_path = "config.toml"
    else:
        if len(sys.argv) == 3 and sys.argv[1] == "--config":
            config_path = sys.argv[2]
        else:
            raise SystemExit(f"Usage: {sys.argv[0]} [--config <path_to_config.toml>]")

    if not os.path.exists(config_path):
        raise Exception(f"Cannot find config file `{config_path}`")

    with open(config_path, "rb") as f:
        try:
            config = tomllib.load(f)
        except Exception as e:
            raise Exception(f"Cannot parse config:\n{e}")

    return config


def check_keys(config: dict, keys: list[str]):
    """Check whether the given keys are correctly defined in the config."""
    for key in keys:
        key_as_list = key.split(".")
        inner_config = config
        for k in key_as_list:
            if not k in inner_config:
                raise Exception(f"Missing key `{key}` in config")
            inner_config = inner_config[k]


def read_secrets() -> dict:
    """Reads the `secrets.toml` file from the default path."""
    path = "secrets.toml"
    if not os.path.exists(path):
        raise Exception(f"Cannot find secrets file `{path}`")
    with open(path, "rb") as f:
        try:
            secrets = tomllib.load(f)
        except Exception as e:
            raise Exception(f"Cannot parse secrets:\n{e}")
    return secrets
