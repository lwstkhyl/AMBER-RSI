"""Read the one editable configuration, or a run's immutable snapshot."""
import json
import os
from pathlib import Path
from string import Template

PACKAGE = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get('LDPRED_CONFIG', PACKAGE / 'config.json')).resolve()
    config = json.loads(path.read_text())
    paths = config['paths']
    variables = dict(os.environ, package_root=str(PACKAGE))
    for _ in range(len(paths) + 1):
        variables.update({k: v for k, v in paths.items() if isinstance(v, str)})
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = os.path.expanduser(Template(value).safe_substitute(variables))
            elif isinstance(value, list):
                paths[key] = [os.path.expanduser(Template(v).safe_substitute(variables)) for v in value]
    return config


CONFIG = load_config()
