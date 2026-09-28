import os
from functools import lru_cache


@lru_cache(maxsize=256)
def _scan_folder(folder: str) -> tuple:
    """Names in `folder`, sorted, cached per folder.

    A simulation folder is read once per timestamp by each of the two loaders.
    Globbing per call meant 28 directory scans of a 301-entry folder to read 300
    files -- about 8,400 metadata operations per folder and 8.6 million per
    generation of 1024, which is what made the analysis step latency-bound on
    GPFS rather than bandwidth-bound. One scan per folder removes that.

    Analysis runs after every simulation in the folder is complete and the
    incomplete ones have been pruned, so the listing does not change underneath
    the cache.
    """
    try:
        return tuple(sorted(os.listdir(folder)))
    except OSError:
        return ()


def list_simulation_files(folder, suffix: str):
    """Paths in `folder` whose name ends with `suffix`, in seed order.

    Names are `<exp_group>_<exp_name>_<seed>_<timestamp>.<type>.json` with a
    zero-padded four-digit seed, so a lexical sort is a sort by seed.
    """
    from pathlib import Path

    folder = Path(folder)
    return [folder / name for name in _scan_folder(str(folder)) if name.endswith(suffix)]


from typing import Dict
import re


class FileParser:
    @staticmethod
    def parse_simulation_file(filename: str, file_type: str) -> Dict[str, str]:
        """Parse simulation filename to extract experiment info

        Args:
            filename: Format <exp_group>_<exp_name>_<seed>_<timestamp>.<file_type>.json
            file_type: Type of file (e.g., 'CELLS' or 'LOCATIONS')

        Returns:
            Dictionary containing parsed components
        """
        pattern = rf"(.+?)_(.+?)_(\d+)_(\d+)\.{file_type}\.json"
        match = re.match(pattern, filename)
        if match:
            return {
                "exp_group": match.group(1),
                "exp_name": match.group(2),
                "seed": match.group(3),
                "timestamp": match.group(4),
            }
        return {}
