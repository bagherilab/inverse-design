import subprocess
from pathlib import Path
import concurrent.futures
import logging
from tqdm import tqdm
import sys
import os
import multiprocessing as mp


def run_single_simulation(input_file: Path, output_base_dir: Path, jar_path: str) -> bool:
    """Run a single ARCADE simulation

    Args:
        input_file: Path to input XML file
        output_base_dir: Base directory for outputs
        jar_path: Path to arcade_v3.jar

    Returns:
        bool: True if simulation completed successfully
    """
    output_dir = f"{output_base_dir}/inputs/{input_file.stem}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # try:
    result = subprocess.run(
        # ["java", "-Djava.awt.headless=true", "-jar", str(jar_path), "patch", str(input_file), str(output_dir)],
        [
            "java",
            "-Djava.awt.headless=true",
            "-cp",
            str(jar_path),
            "arcade.core.ARCADE",
            "patch",
            str(input_file),
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return True
    # except subprocess.CalledProcessError as e:
    #    logging.error(f"Error running simulation for {input_file.name}")
    #    logging.error(f"Error message: {e.stderr}")
    #    return False


def get_input_number(filename):
    """Extract the numerical value from input filename."""
    try:
        return int(filename.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def run_simulations(
    input_dir: str = "perturbed_inputs",
    output_dir: str = "simulation_results",
    jar_path: str = "arcade_v3.jar",
    max_workers: int = 4,
    running_index: list[int] = [None],
) -> None:
    """Run ARCADE simulations for all input files in parallel

    Args:
        input_dir: Directory containing input XML files
        output_dir: Directory for simulation outputs
        jar_path: Path to arcade_v3.jar
        max_workers: Maximum number of parallel simulations
        running_index: List of indices of the simulations to run
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # Relative path resolves against the process cwd, which is inside the
            # read-only image when running under apptainer --pwd. DED_LOG_DIR lets
            # callers point it at writable storage.
            logging.FileHandler(
                os.path.join(os.environ.get("DED_LOG_DIR", "."), "simulation_run.log")
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    jar_path = Path(jar_path)

    if not jar_path.exists():
        raise FileNotFoundError(f"ARCADE jar file not found at {jar_path}")
    print(input_dir)
    input_files = sorted(input_dir.glob("*.xml"))
    if not input_files:
        raise FileNotFoundError(f"No XML files found in {input_dir}")

    input_files = sorted(input_dir.glob("input_*"), key=get_input_number)
    if running_index == [None]:
        running_index = list(range(len(input_files)))
    else:
        running_index = [int(i) for i in running_index]
        input_files = [f for f in input_files if get_input_number(f) in running_index]
    if not input_files:
        raise FileNotFoundError(f"No XML files found with index in {input_dir.name}")

    logging.info(f"Found {len(input_files)} input files with index in {input_dir.name}")
    logging.info(f"Running simulations with {max_workers} parallel workers")

    output_dir.mkdir(exist_ok=True, parents=True)

    successful = 0
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_single_simulation, input_file, output_dir, jar_path): input_file
            for input_file in input_files
        }

        with tqdm(total=len(input_files), desc="Running simulations") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"Simulation failed for {input_file.name}: {str(e)}")
                    failed += 1
                pbar.update(1)

    logging.info(f"Completed simulations: {successful} successful, {failed} failed")


if __name__ == "__main__":
    metric = "doub_time"
    run_simulations(
        input_dir=f"TEST/ARCADE_INPUT/",
        output_dir=f"TEST/ARCADE_OUTPUT/",
        # jar_path="models/arcade-logging-necrotic.jar",
        jar_path="models/arcade-share_space.jar",
        max_workers=int(mp.cpu_count() / 2),
        # running_index=[i for i in range(1, 325)],
    )
