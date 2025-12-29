"""
Enter your machine/cluster specific paths to data and outputs here.
This file will be committed but changes left untracked. 
"""

from pathlib import Path
import os


def _load_dotenv():
    repo_root = Path(__file__).resolve().parents[2]
    dotenv_path = repo_root / ".env"
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

WORK_DIR = os.environ.get("WORK_DIR", "out")

TRAINING_DATA_PATH = "test"

OUTPUT_PATHS = str(Path(WORK_DIR))
