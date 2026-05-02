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
