#!/usr/bin/env python3
"""Select five seeds for g0--g3 and ten seeds for reported generation g4.

The prepare driver first analyses the raw generation that just finished and
then creates the next generation in the same process.  Therefore g4 is created
when three generations (g0--g2) already have summary CSVs and g3 raw output is
about to be analysed.  Switching at four analysed generations is one prepare
too late.
"""

import json
import sys
from pathlib import Path


FIVE = "sample_inputs/sample_combined_v3_5seed.xml"
TEN = "sample_inputs/sample_combined_v3.xml"
G4_PREPARE_THRESHOLD = 3


def analysed_generations(run_root, n_iterations):
    return sum(
        1
        for generation in range(n_iterations)
        if (run_root / f"iter_{generation}" / "final_metrics.csv").is_file()
    )


def update_config(config_path):
    config = json.loads(config_path.read_text())
    run_root = Path(config["base_output_dir"]) / config["run_name"]
    done = analysed_generations(run_root, config["n_iterations"])
    template = TEN if done >= G4_PREPARE_THRESHOLD else FIVE
    if config.get("template_path") != template:
        config["template_path"] = template
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    return done, template


def main():
    config_path = Path(sys.argv[1])
    done, template = update_config(config_path)
    print(f"analysed_generations={done} template={template}")


if __name__ == "__main__":
    main()
