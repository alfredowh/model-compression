from pathlib import Path
import glob
import re
from typing import List


def increment_path(path):
    path = Path(path)
    if not path.exists():
        return path
    else:
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return Path(f"{path}{n}")  # update path


def calc_total_ratio(ratios: List[float]) -> float:
    x = 1
    for r in ratios:
        x = x - (x * r)
    return 1 - x
