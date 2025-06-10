__all__ = ["get_root_directory", "get_default_results_directory"]

from pathlib import Path


def get_root_directory():
    return Path(__file__).parent.parent.parent


def get_default_results_directory():
    return get_root_directory() / "results"
