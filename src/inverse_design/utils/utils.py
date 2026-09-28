# utils.py
import logging
import os
from typing import List, Dict
import pandas as pd
import numpy as np
import zipfile
import shutil


def remove_outliers(data, iqr_multiplier):
    if len(data) == 0:
        return data, []

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Check if data is a Series or DataFrame and adjust axis accordingly
    if isinstance(data, pd.Series):
        keep_mask = ~((data < lower_bound) | (data > upper_bound))
    else:  # DataFrame
        keep_mask = ~((data < lower_bound) | (data > upper_bound)).any(axis=1)

    # Get outlier indices (those not in keep_mask)
    outlier_indices = data.index[~keep_mask].tolist()

    # Filtered data
    data_filtered = data[keep_mask]

    return data_filtered, outlier_indices


def map_capillary_density_to_spacings(
    capillary_density, radius_bound, side_length, width_to_length_ratio=1.2
):
    """
    Map capillary density to length and width spacings for a hexagonal grid-source layout,
    ensuring length_spacing < width_spacing.

    Parameters:
    capillary_density (float): Target capillary density (capillaries/mm^2)
    radius_bound (float): Simulation radius + margin
    side_length (float): Length of one side of the triangular lattice (in microns)
    width_to_length_ratio (float): Ratio of width_spacing to length_spacing (>1)

    Returns:
    tuple: (length_spacing, width_spacing) in microns
    """
    MICRON_to_MM = 1e-3
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2
    area = ((length + 1) / 2 * side_length) * (width * side_length)
    area = area * MICRON_to_MM**2  # convert to mm^2

    num_capillaries = capillary_density * area
    # Check if num_capillaries is an integer and a product of two integers
    factor_pairs = _all_factor_pairs(num_capillaries)
    factor_pairs = factor_pairs[1:]

    length_spacing = ((length * width) / (capillary_density * width_to_length_ratio * area)) ** 0.5
    width_spacing = width_to_length_ratio * length_spacing
    return length_spacing, width_spacing


def _all_factor_pairs(n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")

    pairs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs


def get_samples_data(
    param_metrics_distances_results: List[Dict],
    model_type: str,
    save_path: str = "all_samples_metrics.csv",
):
    """Save all samples data to a CSV file

    Args:
        param_metrics_distances_results: List of dictionaries containing all samples metrics
        model_type: Type of model used (e.g., "BDM", "ARCADE")
        save_path: Path where to save the CSV file
    """
    # Define columns based on model type
    if model_type == "BDM":
        columns = [
            "proliferate",
            "death",
            "migrate",
            "cell_density",
            "time_to_eq",
            "distance",
            "accepted",
        ]
    elif model_type == "ARCADE":
        columns = [
            "division",
            "death",
            "motility",
            "adhesion",
            "cell_density",
            "cluster_size",
            "distance",
            "accepted",
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    all_samples_df = pd.DataFrame(param_metrics_distances_results, columns=columns)
    all_samples_df.to_csv(save_path, index=False)

    return all_samples_df


def zip_directory(source_dir, zip_path, include_root_folder=True):
    """
    Zip a directory and all its contents.

    Args:
        source_dir (str): Path to the directory to zip
        zip_path (str): Path where the zip file will be created
        include_root_folder (bool): If True, includes the root folder name in the zip.
                                   If False, zips only the contents.

    Examples:
        # This will create a zip with folder structure: my_folder/file1.txt
        zip_directory('/path/to/my_folder', 'output.zip', include_root_folder=True)

        # This will create a zip with structure: file1.txt (no root folder)
        zip_directory('/path/to/my_folder', 'output.zip', include_root_folder=False)
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if include_root_folder:
            # Include the root folder name in the archive
            root_folder_name = os.path.basename(source_dir.rstrip("/\\"))
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate relative path from parent of source_dir
                    arcname = os.path.join(root_folder_name, os.path.relpath(file_path, source_dir))
                    zipf.write(file_path, arcname)
        else:
            # Don't include root folder, zip contents directly
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate relative path to maintain subdirectory structure
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)

    print(f"Created zip file: {zip_path}")
    return zip_path


def unzip_to_named_folder(zip_path, extract_to=None):
    """
    Unzip a file to a folder with the same name as the zip file (without .zip extension).

    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Base directory to extract to. If None, uses zip file's directory.

    Returns:
        str: Path to the extracted folder

    Example:
        # If zip_path is '/downloads/results.zip'
        # This will extract to '/downloads/results/'
        unzip_to_named_folder('/downloads/results.zip')
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    # Get zip filename without extension
    zip_filename = os.path.basename(zip_path)
    folder_name = os.path.splitext(zip_filename)[0]  # Remove .zip extension

    # Create extraction path
    extract_path = os.path.join(extract_to, folder_name)

    # Create directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Extract zip file
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_path)

    print(f"Extracted {zip_path} to {extract_path}")
    return extract_path


def clean_zip_structure(extracted_folder):
    """
    Clean up nested folder structure if the zip contained only one root folder.
    This moves contents up one level if there's unnecessary nesting.

    Args:
        extracted_folder (str): Path to the extracted folder

    Returns:
        str: Path to the cleaned folder
    """
    contents = os.listdir(extracted_folder)

    # If there's only one item and it's a directory, move its contents up
    if len(contents) == 1:
        single_item = os.path.join(extracted_folder, contents[0])
        if os.path.isdir(single_item):
            print(f"Found nested folder structure, flattening...")

            # Create temporary directory
            temp_dir = extracted_folder + "_temp"

            # Move the nested folder to temp location
            shutil.move(single_item, temp_dir)

            # Remove the now-empty extracted folder
            os.rmdir(extracted_folder)

            # Rename temp directory to the original name
            shutil.move(temp_dir, extracted_folder)

            print(f"Flattened folder structure in {extracted_folder}")

    return extracted_folder


def zip_and_prepare_for_clean_extraction(source_dir, zip_path):
    """
    Zip directory in a way that extracts cleanly to a folder with the zip's name.

    Args:
        source_dir (str): Directory to zip
        zip_path (str): Output zip file path

    Returns:
        str: Path to created zip file
    """
    # Zip without including root folder to avoid double nesting
    return zip_directory(source_dir, zip_path, include_root_folder=False)


def extract_zip_cleanly(zip_path, extract_to=None):
    """
    Extract zip file to a clean folder structure with the zip file's name.

    Args:
        zip_path (str): Path to zip file
        extract_to (str): Where to extract (optional)

    Returns:
        str: Path to extracted folder
    """
    # Extract to named folder
    extracted_path = unzip_to_named_folder(zip_path, extract_to)

    # Clean up any unnecessary nesting
    cleaned_path = clean_zip_structure(extracted_path)

    return cleaned_path


def archive_directory(source_dir, archive_path=None, *, level=3, threads=0, exclude=None):
    """Archive a directory quickly, preferring multithreaded zstd over Python's zipfile.

    `zip_directory` uses `zipfile` with ZIP_DEFLATED, which is single-threaded and
    pays Python-level overhead per member. On an ARCADE generation (~300k small
    JSON files) that dominates the whole upload step -- the transfer itself is one
    object and takes minutes.

    `tar` piped through `zstd -T0` runs the compressor across all cores and keeps
    the per-file work in C, which is typically an order of magnitude faster at a
    comparable ratio.

    Falls back to `zip_directory` when `tar`/`zstd` are unavailable, so callers
    always get an archive.

    Args:
        source_dir: Directory to archive.
        archive_path: Output path. Defaults to `<source_dir>.tar.zst`
            (or `<source_dir>.zip` on fallback).
        level: zstd compression level. 3 is the speed/ratio sweet spot here.
        threads: zstd worker threads; 0 means one per core.
        exclude: glob patterns to leave out, matched against the file name.
            Used to keep raw ARCADE `*.CELLS.json` / `*.LOCATIONS.json` out
            of an archive: they are regenerable from the input XML and the
            pinned ARCADE version, and they dominate the size by ~4 orders
            of magnitude.

    Returns:
        Path to the archive that was created.
    """
    import fnmatch
    import subprocess

    source_dir = str(source_dir).rstrip("/")
    parent = os.path.dirname(source_dir) or "."
    name = os.path.basename(source_dir)
    patterns = list(exclude or [])
    excl_args = [arg for pattern in patterns for arg in ("--exclude", pattern)]

    if shutil.which("tar"):
        # zstd is the fastest option but is absent from the analysis container,
        # so gzip is the realistic path. Either way the per-file work stays in
        # C; Python's zipfile pays interpreter overhead per member, which is
        # what makes it unusable on an ARCADE generation of ~300k small files.
        if shutil.which("zstd"):
            out = archive_path or f"{source_dir}.tar.zst"
            compressor = ["-I", f"zstd -{level} -T{threads}"]
        else:
            out = archive_path or f"{source_dir}.tar.gz"
            compressor = ["-z"]
        cmd = ["tar", *excl_args, *compressor, "-cf", out, "-C", parent, name]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return out
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            logging.warning("tar archiving failed (%s); falling back to zip", stderr.strip())

    if not patterns:
        return zip_directory(
            source_dir, archive_path or f"{source_dir}.zip", include_root_folder=True
        )

    # The zip fallback has to honour `exclude` too. Silently ignoring it would
    # archive the raw ARCADE output the caller asked to leave out -- tens of
    # gigabytes per generation.
    out = archive_path or f"{source_dir}.zip"
    root_folder_name = os.path.basename(source_dir)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(source_dir):
            for file in files:
                if any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.join(
                    root_folder_name, os.path.relpath(file_path, source_dir)
                )
                zipf.write(file_path, arcname)
    return out
