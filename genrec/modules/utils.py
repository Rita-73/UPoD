"""Config parsing and logging helpers."""
import argparse
import gin
import logging
import os
from datetime import datetime
from typing import Optional, Set


def setup_logger(save_dir: str, name: str = "train") -> logging.Logger:
    """Write logs to both a file under ``save_dir`` and stdout."""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(save_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


_RUN_CONTEXT: dict = {}


def get_run_split(default: str = "unknown") -> str:
    """Return --split as resolved by the most recent parse_config() call."""
    return _RUN_CONTEXT.get("split", default)


def _resolve_gin_include_path(include_rel: str, config_dir: str) -> str:
    candidates = [
        include_rel,
        os.path.normpath(os.path.join(config_dir, include_rel)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"Unable to resolve gin include '{include_rel}' from {config_dir}"
    )


def _expand_gin_includes(config_path: str, _visited: Optional[Set[str]] = None) -> str:
    if _visited is None:
        _visited = set()

    abs_path = os.path.abspath(config_path)
    if abs_path in _visited:
        raise ValueError(f"Circular gin include detected: {abs_path}")
    _visited.add(abs_path)

    config_dir = os.path.dirname(abs_path)
    with open(abs_path, "r") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('include "') and stripped.endswith('"'):
            include_rel = stripped[len('include "'):-1]
            include_path = _resolve_gin_include_path(include_rel, config_dir)
            out.append(_expand_gin_includes(include_path, _visited))
        else:
            out.append(line)
    return "".join(out)


def parse_config():
    """Parse the gin config file with {split} placeholder support."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    parser.add_argument(
        "--split",
        type=str,
        default="beauty",
        help="Dataset split. Replaces {split} in config.",
    )
    parser.add_argument(
        "--gin",
        action="append",
        default=[],
        help="Gin parameter overrides (can be specified multiple times).",
    )
    args = parser.parse_args()

    config_content = _expand_gin_includes(args.config_path)
    if args.split:
        config_content = config_content.replace("{split}", args.split)

    gin.parse_config(config_content)
    if args.gin:
        if args.split:
            args.gin = [g.replace("{split}", args.split) for g in args.gin]
        gin.parse_config(args.gin)

    _RUN_CONTEXT["split"] = args.split
    _RUN_CONTEXT["config_path"] = args.config_path
    return args
