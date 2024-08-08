"""Rohith-AI-839 file for ensuring the package is executable
as `rohith-ai-839` and `python -m rohith_ai_839`
"""
from pathlib import Path

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs):
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()
