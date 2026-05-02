import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from inverse_design.plotting.colormap import metric_color, sir_compartment_color
from inverse_design.plotting.theme import apply_publication_style

compartment_to_color = {
    "I": sir_compartment_color("I"),
    "R": sir_compartment_color("R"),
    "S": sir_compartment_color("S"),
}

# SIR summary statistics columns (fixed order) and matplotlib mathtext axis labels.
SIR_METRIC_COLS: tuple[str, ...] = (
    "peak_I",
    "time_to_peak",
    "growth_rate",
    "area_I",
    "final_R",
)
SIR_METRIC_DISPLAY_LABELS: tuple[str, ...] = (
    r"$I_\mathrm{peak}$",
    r"$t_\mathrm{peak}$",
    r"$g_I$",
    r"$A_I$",
    r"$R_\mathrm{final}$",
)


def load_histories_from_dir(hist_dir, iteration, n_samples=100, seed=42):
    """
    hist_dir: path to .../n3_t365_l30/n512/
    Scans hist_dir/iter_{iteration}/history/ for history_{i}.csv files.
    Randomly samples n_samples indices (seeded). Returns list of DataFrames.
    Each CSV has columns: time, S, I, R, total_infected.
    """
    history_dir = Path(hist_dir) / f"iter_{iteration}" / "history"
    history_files = []

    for path in history_dir.glob("history_*.csv"):
        try:
            idx = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        history_files.append((idx, path))

    history_files = sorted(history_files, key=lambda item: item[0])
    if not history_files:
        return []

    available_indices = np.array([idx for idx, _ in history_files])
    file_by_index = {idx: path for idx, path in history_files}

    rng = np.random.default_rng(seed)
    sample_size = min(n_samples, len(available_indices))
    sampled_indices = rng.choice(available_indices, size=sample_size, replace=False)

    histories = []
    for idx in sampled_indices:
        histories.append(pd.read_csv(file_by_index[int(idx)]))

    return histories


def plot_sensitivity_heatmap(
    params_df,
    metrics_df,
    ax,
    *,
    annot_fontsize=10,
    tick_labelsize=9,
    metric_labelsize=None,
    cbar_tick_labelsize=9,
    cbar_ylabel_fontsize=10,
    x_tick_rotation=45,
    x_tick_ha="right",
    cbar_ticks=None,
    cbar_label_position="right",
    cbar_labelpad=None,
):
    """
    params_df: DataFrame of shape (n_particles, 4) columns=[PI, PR, IIF, ISF]
    metrics_df: DataFrame with columns [peak_I, time_to_peak, growth_rate, area_I, final_R]
    Computes Spearman rho between each param and each metric.
    Renders as seaborn heatmap on ax:
      - cmap='RdBu_r', vmin=-1, vmax=1
      - annot=Spearman rho (same as fill color), rounded 2 decimals
      - linewidths=0.5, linecolor='white'
      - colorbar label = Spearman ρ (mathtext)
      - row labels: PI, PR, IIF, ISF
      - col labels: mathtext I_peak, t_peak, g_I, A_I, R_final
    """
    if metric_labelsize is None:
        metric_labelsize = tick_labelsize

    param_cols = ["PI", "PR", "IIF", "ISF"]
    metric_cols = list(SIR_METRIC_COLS)
    metric_labels = list(SIR_METRIC_DISPLAY_LABELS)

    min_rows = min(len(params_df), len(metrics_df))
    params_aligned = params_df.iloc[:min_rows][param_cols].reset_index(drop=True)
    metrics_aligned = metrics_df.iloc[:min_rows][metric_cols].reset_index(drop=True)
    aligned = pd.concat([params_aligned, metrics_aligned], axis=1).dropna()

    params_clean = aligned[param_cols]
    metrics_clean = aligned[metric_cols]

    rho_matrix = np.zeros((len(param_cols), len(metric_cols)))
    for i, param_col in enumerate(param_cols):
        for j, metric_col in enumerate(metric_cols):
            rho, _ = spearmanr(params_clean[param_col], metrics_clean[metric_col])
            rho_matrix[i, j] = rho

    rho_df = pd.DataFrame(rho_matrix, index=param_cols, columns=metric_labels)
    annot = np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.2f}")(rho_matrix)

    heatmap = sns.heatmap(
        rho_df,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": annot_fontsize},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={},
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        metric_labels,
        rotation=x_tick_rotation,
        ha=x_tick_ha,
        fontsize=metric_labelsize,
    )
    ax.set_yticklabels(param_cols, rotation=0, fontsize=tick_labelsize)
    ax.tick_params(axis="x", which="major", labelsize=metric_labelsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize)

    cbar = heatmap.collections[0].colorbar
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=cbar_tick_labelsize)
    cbar.ax.set_ylabel(r"Spearman $\rho$", fontsize=cbar_ylabel_fontsize)
    cbar.ax.yaxis.set_label_position(cbar_label_position)
    if cbar_labelpad is not None:
        cbar.ax.yaxis.labelpad = cbar_labelpad

    return rho_df


def visualize_results(
    history_paths, statistics_path, target_values_path, compartment="I", save_path=None, ax=None
):
    """Visualize simulation results with overlapping particle histories, highlighting the best particle.

    Args:
        history_paths (list): List of paths to history CSV files
        statistics_path (str): Path to statistics CSV file
        target_values_path (str): Path to target values CSV file
        compartment (str): Which compartment to plot ('S', 'I', or 'R')
        save_path (str, optional): Path to save the plot
    """

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    # Find the best particle
    best_result = find_best_params(statistics_path, target_values_path)
    best_particle_idx = best_result["best_particle_idx"]

    # Load target values for labeling
    target_values = pd.read_csv(target_values_path).iloc[0]

    # Read and plot each particle's history
    count = 0
    max_count = 100
    for i, history_path in enumerate(history_paths):
        try:
            history = pd.read_csv(history_path)
            # Determine color and line properties based on whether this is the best particle
            if i == best_particle_idx:
                color = compartment_to_color[compartment]
                alpha = 1.0
                linewidth = 2.5
                zorder = 10  # Ensure best particle is plotted on top
                label = f'Best Particle {i} (MSE: {best_result["best_mse"]:.6f})'
                count += 1
                ax.plot(
                    history["time"],
                    history[compartment],
                    linestyle="--",
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=zorder,
                    label=label if label else None,
                )
            else:
                color = "gray"
                alpha = 0.2
                linewidth = 0.3
                zorder = 1
                label = f"Particle {i}" if i < 3 else ""  # Only label first few for legend clarity
                count += 1
            if count > max_count:
                continue
            # Plot the selected compartment
            ax.plot(
                history["time"],
                history[compartment],
                linestyle="--",
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
                label=label if label else None,
            )

        except Exception as e:
            print(f"Error reading {history_path}: {e}")
            continue

    # Add target value annotations based on compartment
    if compartment == "I":
        # For infected: show peak_I and time_to_peak
        peak_I = target_values["peak_I"]
        time_to_peak = target_values["time_to_peak"]

        ax.plot(
            time_to_peak,
            peak_I,
            marker="*",
            color=compartment_to_color[compartment],
            markersize=15,
            zorder=6,
        )

    elif compartment == "R":
        # For recovered: show final_R
        final_R = target_values["final_R"]
        ax.plot(
            ax.get_xlim()[1],
            final_R,
            marker="*",
            color=compartment_to_color[compartment],
            markersize=15,
            zorder=6,
        )

    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    # ax.spines['top'].set_linewidth(1.0)
    # ax.spines['right'].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)

    # Labels and formatting based on compartment
    compartment_names = {"S": "Susceptible", "I": "Infected", "R": "Recovered"}
    # ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    # ax.set_ylabel(f'{compartment_names[compartment]} Fraction', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])
    # ax.set_xticks([0, 20, 40, 60, 80, 100])

    if created_fig:
        plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if created_fig:
            plt.close(fig)  # Close the figure to free memory
    elif created_fig:
        plt.show()

    return ax


def visualize_results_best_histories(
    true_history_path,
    best_history_paths,
    target_values_path,
    compartments=["I", "R", "S"],
    save_path=None,
    ax=None,
):
    """Visualize simulation results with true parameters and best simulated histories.

    Args:
        true_history_path (str): Path to true parameters history CSV file
        best_history_paths (list): List of paths to best simulated parameters history CSV files
        target_values_path (str): Path to target values CSV file
        compartment (str): Which compartment to plot ('S', 'I', or 'R')
        save_path (str, optional): Path to save the plot
    """

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    def _load_history(history_source):
        if isinstance(history_source, pd.DataFrame):
            return history_source
        return pd.read_csv(history_source)

    # Load target values for labeling
    target_values = pd.read_csv(target_values_path).iloc[0]

    # Plot true parameters history
    try:
        if isinstance(true_history_path, (list, tuple)):
            true_histories = true_history_path
        else:
            true_histories = [true_history_path]

        for i, history_source in enumerate(true_histories):
            true_history = _load_history(history_source)
            is_bold_reference = i == len(true_histories) - 1
            for compartment in compartments:
                ax.plot(
                    true_history["time"],
                    true_history[compartment],
                    color=compartment_to_color[compartment],
                    alpha=1.0 if (created_fig or is_bold_reference) else 0.5,
                    linewidth=1.5 if (created_fig or is_bold_reference) else 1.2,
                    zorder=10 if (created_fig or is_bold_reference) else 4,
                    linestyle="--",
                )
    except Exception as e:
        print(f"Error reading true history {true_history_path}: {e}")

    # Plot best simulated parameters histories (multiple CSV files for stochasticity)
    for i, best_history_path in enumerate(best_history_paths):
        try:
            best_history = _load_history(best_history_path)
            for compartment in compartments:
                ax.plot(
                    best_history["time"],
                    best_history[compartment],
                    color=compartment_to_color[compartment],
                    alpha=1.0 if not created_fig else 0.1,
                    linewidth=1.5 if not created_fig else 1,
                    zorder=8 if not created_fig else 5,
                    label="Best Simulated" if i == 0 else "",
                )

        except Exception as e:
            print(f"Error reading best history {best_history_path}: {e}")
            continue
        if 0:
            # Add target value annotations based on compartment
            if compartment == "I":
                # For infected: show peak_I and time_to_peak
                peak_I = target_values["peak_I"]
                time_to_peak = target_values["time_to_peak"]
                ax.plot(
                    time_to_peak,
                    peak_I,
                    marker="*",
                    color=compartment_to_color[compartment],
                    markersize=15,
                    zorder=15,
                    label="Target",
                )

            elif compartment == "R":
                # For recovered: show final_R
                final_R = target_values["final_R"]
                ax.plot(
                    ax.get_xlim()[1],
                    final_R,
                    marker="*",
                    color=compartment_to_color[compartment],
                    markersize=15,
                    zorder=15,
                    label="Target",
                )

    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)

    # ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Fraction of Population', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)
    # for label_tick in ax.get_xticklabels():
    #    label_tick.set_fontweight('bold')
    # for label_tick in ax.get_yticklabels():
    #    label_tick.set_fontweight('bold')

    # Add legend
    # ax.legend(loc='best', frameon=False)

    # Tight layout for better spacing
    if created_fig:
        plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        if created_fig:
            plt.close(fig)  # Close the figure to free memory
    elif created_fig:
        plt.show()

    return ax


def find_best_params(statistics_path, target_values_path, params_path=None):
    """
    Find the best parameter set based on minimum MSE between normalized particle statistics and target values.

    Args:
        statistics_path (str): Path to the statistics CSV file
        target_values_path (str): Path to the target values CSV file
        params_path (str, optional): Path to the parameters CSV file

    Returns:
        dict: Dictionary containing best particle info, MSE values, and parameters if available
    """
    # Load data
    statistics = pd.read_csv(statistics_path)
    target_values = pd.read_csv(target_values_path)

    # Combine statistics and target values for normalization
    all_data = pd.concat([statistics, target_values], ignore_index=True)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(all_data)

    # Split back into statistics and target
    normalized_statistics = normalized_data[:-1]  # All rows except last
    normalized_target = normalized_data[-1]  # Last row (target values)
    # Calculate MSE for each particle using normalized data
    mse_values = []
    for i in range(len(normalized_statistics)):
        particle_stats = normalized_statistics[i]
        mse = np.mean((particle_stats - normalized_target) ** 2)
        mse_values.append(mse)
    # Find the best particle (minimum MSE)
    best_particle_idx = np.argmin(mse_values)
    best_mse = mse_values[best_particle_idx]

    # Prepare results
    result = {
        "best_particle_idx": best_particle_idx,
        "best_mse": best_mse,
        "all_mse_values": mse_values,
        "best_statistics": statistics.iloc[best_particle_idx].to_dict(),
        "target_values": target_values.iloc[0].to_dict(),
        "scaler": scaler,  # Include scaler for potential future use
    }

    # Load parameters if available
    if params_path:
        try:
            params = pd.read_csv(params_path)
            result["best_parameters"] = params.iloc[best_particle_idx].to_dict()
        except Exception as e:
            print(f"Warning: Could not load parameters from {params_path}: {e}")

    return result


def analyze_best_params_per_iteration(base_dir, n_particles, n_iterations):
    """
    Analyze and find the best parameters for each iteration.

    Args:
        base_dir (str): Base directory path
        n_particles (int): Number of particles
        n_iterations (int): Number of iterations

    Returns:
        list: List of best parameter results for each iteration
    """
    target_values_path = f"{base_dir}/target_values.csv"
    all_results = []

    print("=" * 60)
    print("BEST PARAMETER ANALYSIS")
    print("=" * 60)

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration}:")
        print("-" * 30)

        # Paths for this iteration
        statistics_path = f"{base_dir}/n{n_particles}/iter_{iteration}/statistics.csv"
        params_path = f"{base_dir}/n{n_particles}/iter_{iteration}/params.csv"

        try:
            # Find best params for this iteration
            result = find_best_params(statistics_path, target_values_path, params_path)
            all_results.append(result)

            # Print results
            print(f"Best particle: {result['best_particle_idx']}")
            print(f"Best MSE: {result['best_mse']:.6f}")
            print(f"Best statistics:")
            for key, value in result["best_statistics"].items():
                target_val = result["target_values"][key]
                print(f"  {key}: {value:.6f} (target: {target_val:.6f})")

            if "best_parameters" in result:
                print(f"Best parameters:")
                for key, value in result["best_parameters"].items():
                    print(f"  {key}: {value:.6f}")

        except Exception as e:
            print(f"Error processing iteration {iteration}: {e}")
            all_results.append(None)

    # Find overall best iteration
    valid_results = [r for r in all_results if r is not None]
    if valid_results:
        overall_best_idx = np.argmin([r["best_mse"] for r in valid_results])
        overall_best = valid_results[overall_best_idx]

        print("\n" + "=" * 60)
        print("OVERALL BEST RESULT")
        print("=" * 60)
        print(f"Best iteration: {overall_best_idx}")
        print(f"Best particle: {overall_best['best_particle_idx']}")
        print(f"Best MSE: {overall_best['best_mse']:.6f}")
        if "best_parameters" in overall_best:
            print(f"Best parameters:")
            for key, value in overall_best["best_parameters"].items():
                print(f"  {key}: {value:.6f}")

    return all_results


def plot_joint_distribution_4params_compact(
    prior_samples,
    posterior_samples,
    true_params,
    param_names=None,
    save_path=None,
    axes=None,
    *,
    tick_labelsize=None,
    axis_labelsize=None,
    top_row_xlabel_pad=5,
    bottom_row_xlabel_pad=None,
    ylabel_pad=None,
):
    """
    Alternative compact version with 2x3 layout for better space utilization.
    """

    prior_array = np.asarray(prior_samples)
    posterior_array = np.asarray(posterior_samples)
    true_array = np.asarray(true_params).reshape(-1)

    if param_names is None:
        if hasattr(prior_samples, "columns"):
            param_names = list(prior_samples.columns)
        else:
            param_names = [f"param_{i}" for i in range(prior_array.shape[1])]

    created_fig = axes is None
    if created_fig:
        # Create square subplots with reduced gaps
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))

        # Reduce spacing between subplots
        plt.subplots_adjust(
            left=0.08,  # Left margin
            bottom=0.08,  # Bottom margin
            right=0.95,  # Right margin
            top=0.95,  # Top margin
            wspace=0.15,  # Width spacing between subplots
            hspace=0.25,  # Height spacing between subplots
        )
    else:
        fig = axes.flat[0].figure

    if not created_fig:
        if tick_labelsize is None:
            tick_labelsize = 9
        if axis_labelsize is None:
            axis_labelsize = 10
    else:
        tick_labelsize = tick_labelsize if tick_labelsize is not None else 14
        axis_labelsize = axis_labelsize if axis_labelsize is not None else 16

    # Create pairwise plots
    param_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # Top row  # Bottom row

    prior_alpha = 0.3 if created_fig else 0.25
    posterior_alpha = 0.3 if created_fig else 0.35
    sample_size = 10 if created_fig else 6
    prior_color = "gray" if created_fig else "#aaaaaa"
    posterior_color = "#bb883b"
    true_size = 200 if created_fig else 48
    true_linewidth = 1.5 if created_fig else 0.9

    for idx, (i, j) in enumerate(param_pairs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        # Embedded 2×3: bottom-row axes can paint over the top row's x-label area; draw top row last.
        if not created_fig:
            ax.set_zorder(4 - row)

        # Plot prior samples
        ax.scatter(
            prior_array[:, i],
            prior_array[:, j],
            alpha=prior_alpha,
            label="Prior",
            s=sample_size,
            color=prior_color,
            edgecolors="black",
            linewidths=0.2,
        )

        # Plot posterior samples
        ax.scatter(
            posterior_array[:, i],
            posterior_array[:, j],
            alpha=posterior_alpha,
            label="Posterior",
            s=sample_size,
            color=posterior_color,
            edgecolors="black",
            linewidths=0.2,
        )

        # Plot true parameters with border
        ax.scatter(
            true_array[i],
            true_array[j],
            facecolor="#bb883b",
            edgecolor="black",
            marker="*",
            s=true_size,
            label="True",
            zorder=10,
            linewidth=true_linewidth,
        )

        # Style the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=1.5)
        # Labels (embedded multi-row: extra x label pad on top row so names are not clipped)
        if not created_fig and row == 0:
            ax.set_xlabel(param_names[i], fontsize=axis_labelsize, labelpad=top_row_xlabel_pad)
            xl = ax.xaxis.get_label()
            xl.set_zorder(30)
            xl.set_clip_on(False)
        else:
            if bottom_row_xlabel_pad is None:
                ax.set_xlabel(param_names[i], fontsize=axis_labelsize)
            else:
                ax.set_xlabel(
                    param_names[i],
                    fontsize=axis_labelsize,
                    labelpad=bottom_row_xlabel_pad,
                )
        if ylabel_pad is None:
            ax.set_ylabel(param_names[j], fontsize=axis_labelsize)
        else:
            ax.set_ylabel(param_names[j], fontsize=axis_labelsize, labelpad=ylabel_pad)
        if not created_fig:
            ax.tick_params(axis="x", labelbottom=True)
        # ax.grid(True, alpha=0.3)

        # Force square aspect ratio
        # ax.set_aspect('equal', adjustable='box')
    if created_fig:
        plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if created_fig:
            plt.close(fig)
    elif created_fig:
        plt.show()

    return axes


def analyze_parameter_errors(
    statistics_paths,
    target_values_path=None,
    iteration_labels=None,
    save_path=None,
    ax=None,
    n_particles=None,
    n_iterations=None,
    *,
    show_swarm=True,
    tick_labelsize=9,
    axis_labelsize=10,
    metric_labelsize=None,
    x_tick_rotation=45,
    x_tick_ha="right",
):
    """Analyze parameter estimation errors and create box (+ optional swarm) or violin plots.

    Args:
        statistics_paths (list or path): List of paths to statistics CSV files (one per
            iteration), or a base n{particles} directory when n_iterations is supplied.
        target_values_path (str, optional): Path to target values CSV file
        iteration_labels (list, optional): Labels for each iteration. If None, uses "Iter 1", "Iter 2", etc.
        save_path (str, optional): Path to save the plot
        ax (matplotlib.axes.Axes, optional): Existing axis. When supplied, render violin-only.
        n_particles (int, optional): Number of particles for base-directory call style.
        n_iterations (int, optional): Number of iterations for base-directory call style.
        show_swarm (bool): If True (default), overlay jittered points on the embedded violins.
    """
    if metric_labelsize is None:
        metric_labelsize = tick_labelsize

    if isinstance(statistics_paths, (str, Path)) and n_iterations is not None:
        base_dir = Path(statistics_paths)
        if not (base_dir / "iter_0").exists() and n_particles is not None:
            base_dir = base_dir / f"n{n_particles}"
        statistics_paths = [
            base_dir / f"iter_{iteration}" / "statistics.csv" for iteration in range(n_iterations)
        ]
        if target_values_path is None:
            candidate = base_dir / "target_values.csv"
            target_values_path = (
                candidate if candidate.exists() else base_dir.parent / "target_values.csv"
            )

    # Load target values
    target_values = pd.read_csv(target_values_path).iloc[0]

    # Set default iteration labels if not provided
    if iteration_labels is None:
        iteration_labels = [f"Iter {i+1}" for i in range(len(statistics_paths))]

    # Load statistics from all iterations and calculate errors
    all_errors = []
    metrics = list(SIR_METRIC_COLS)

    for i, stats_path in enumerate(statistics_paths):
        statistics = pd.read_csv(stats_path)

        # Calculate error percentages for each metric
        iteration_errors = {}
        for metric in metrics:
            target_value = target_values[metric]
            simulated_values = statistics[metric]

            # Calculate percentage error: (simulated - target) / abs(target) * 100
            error_percentages = ((simulated_values - target_value) / abs(target_value)) * 100
            iteration_errors[metric] = error_percentages

        # Add iteration label
        iteration_errors["iteration"] = iteration_labels[i]
        all_errors.append(pd.DataFrame(iteration_errors))

    # Combine all iterations
    combined_errors = pd.concat(all_errors, ignore_index=True)

    # Create single plot
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    else:
        fig = ax.figure

    # Define color scheme (I-metrics vs R-metric; same reds as act_ratio / symmetry)
    _m_red = metric_color("act_ratio")
    _m_green = metric_color("symmetry")
    metrics_to_colors = {
        "peak_I": _m_red,
        "time_to_peak": _m_red,
        "growth_rate": _m_red,
        "area_I": _m_red,
        "final_R": _m_green,
    }

    if not created_fig:
        n_metrics = len(metrics)
        g0_label = iteration_labels[0]
        g4_label = iteration_labels[-1]
        g0_data = combined_errors[combined_errors["iteration"] == g0_label]
        g4_data = combined_errors[combined_errors["iteration"] == g4_label]
        positions = np.arange(1, n_metrics + 1)
        violin_width = 0.26
        offset = 0.20  # g0 left of centre, g4 right

        rng = np.random.default_rng(42) if show_swarm else None

        _BROWN = "#8B4513"
        for j, metric in enumerate(metrics):
            g0_values = np.clip(g0_data[metric].dropna().values, -150, 150)
            g4_values = np.clip(g4_data[metric].dropna().values, -150, 150)
            mc = _BROWN

            pos_g0 = positions[j] - offset
            pos_g4 = positions[j] + offset

            vp = ax.violinplot(
                [g0_values, g4_values],
                positions=[pos_g0, pos_g4],
                widths=violin_width,
                vert=True,
                showmeans=False,
                showextrema=False,
                showmedians=True,
            )
            for body, fc, ed in zip(
                vp["bodies"],
                ["#aaaaaa", mc],
                ["#666666", "#333333"],
            ):
                body.set_facecolor(fc)
                body.set_edgecolor(ed)
                body.set_alpha(0.85)
                body.set_linewidth(0.8)

            cmedians = vp.get("cmedians")
            if cmedians is not None:
                cmedians.set_edgecolor("#111111")
                cmedians.set_linewidths(1.4)

            if show_swarm:
                for values, xc, fc in [
                    (g0_values, pos_g0, "#aaaaaa"),
                    (g4_values, pos_g4, mc),
                ]:
                    jitter = rng.uniform(-0.06, 0.06, len(values))
                    ax.scatter(
                        xc + jitter,
                        values,
                        color=fc,
                        alpha=0.35,
                        s=8,
                        edgecolors="black",
                        linewidths=0.25,
                        zorder=6,
                    )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, zorder=2)
        ax.set_ylabel("Error (%)", fontsize=axis_labelsize)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            list(SIR_METRIC_DISPLAY_LABELS),
            fontsize=metric_labelsize,
            rotation=x_tick_rotation,
            ha=x_tick_ha,
        )
        ax.set_ylim(-150, 150)
        ax.set_xlim(0.5, n_metrics + 0.5)
        ax.legend(
            handles=[
                Patch(facecolor="#aaaaaa", edgecolor="#666666", alpha=0.85, label="g0"),
                Patch(facecolor=_BROWN, edgecolor="#333333", alpha=0.85, label="g4"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=tick_labelsize,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="y", which="major", labelsize=tick_labelsize, width=1.0)
        ax.tick_params(axis="x", which="major", labelsize=metric_labelsize, width=1.0)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return combined_errors

    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        """Convert RGB tuple to hex color"""
        return "#%02x%02x%02x" % rgb

    def lighten_color(hex_color, factor):
        """Lighten a hex color by interpolating with white"""
        rgb = hex_to_rgb(hex_color)
        white = (255, 255, 255)
        lightened_rgb = tuple(int(rgb[i] + factor * (white[i] - rgb[i])) for i in range(3))
        return rgb_to_hex(lightened_rgb)

    # Calculate positions for box plots
    n_iterations = len(statistics_paths)
    n_metrics = len(metrics)
    width = 0.8 / n_iterations  # Width of each box

    # Create box + swarm plots
    for i, iteration_label in enumerate(iteration_labels):
        iteration_data = combined_errors[combined_errors["iteration"] == iteration_label]

        for j, metric in enumerate(metrics):
            # Calculate position for this box
            pos = j + 1 + (i - n_iterations / 2 + 0.5) * width

            # Determine color for this iteration and metric
            if i == 0:  # First iteration - gray
                color = "#808080"
            else:  # Iterations 2 to N - gradient from light to dark for each metric
                base_color = metrics_to_colors[metric]
                if n_iterations == 1:
                    color = base_color
                else:
                    # Create gradient: lightest for iteration 2, darkest (original) for last iteration
                    lightness_factor = (
                        0.7 * (n_iterations - 1 - i) / (n_iterations - 2) if n_iterations > 2 else 0
                    )
                    color = lighten_color(base_color, lightness_factor)

            # Create box plot
            box_data = [iteration_data[metric].values]
            box_plot = ax.boxplot(
                box_data,
                positions=[pos],
                widths=width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=4,
                    alpha=0.6,
                ),
            )

            # Add swarm plot overlay
            n_points = len(iteration_data[metric])
            x_jitter = np.random.normal(pos, width * 0.1, n_points)  # Small jitter around position

            ax.scatter(
                x_jitter,
                iteration_data[metric],
                alpha=0.6,
                s=12,
                color=color,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
                label=iteration_label if j == 0 else "",
            )  # Only label once per iteration

    # Add horizontal line at y=0 (no error)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.7, linewidth=1.5, label="No Error")

    # Customize the plot
    # ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel("Error (%)", fontsize=12)

    # Set x-axis ticks and labels
    ax.set_xticks(range(1, n_metrics + 1))
    ax.set_xticklabels(list(SIR_METRIC_DISPLAY_LABELS), fontsize=metric_labelsize)
    ax.set_ylim(-100, 100)
    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Style the axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize, width=1.5)
    ax.tick_params(axis="x", which="major", labelsize=metric_labelsize, width=1.5)

    # Adjust layout
    plt.tight_layout()

    # Print summary statistics
    print("Error Analysis Summary by Iteration:")
    print("=" * 60)

    for iteration_label in iteration_labels:
        print(f"\n{iteration_label}:")
        print("-" * 40)
        iteration_data = combined_errors[combined_errors["iteration"] == iteration_label]

        for metric in metrics:
            mean_error = iteration_data[metric].mean()
            std_error = iteration_data[metric].std()
            median_error = iteration_data[metric].median()

            print(f"  {metric.replace('_', ' ').title()}:")
            print(
                f"    Mean: {mean_error:.2f}%, Std: {std_error:.2f}%, Median: {median_error:.2f}%"
            )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if created_fig:
            plt.close(fig)
    else:
        plt.show()

    return combined_errors


def plot_correlation_heatmap(
    params_path,
    statistics_path,
    correlation_method="pearson",
    param_names=None,
    stat_names=None,
    save_path=None,
):
    """Plot correlation heatmap between parameters and summary statistics.

    Args:
        params_path (str): Path to parameters CSV file
        statistics_path (str): Path to statistics CSV file
        correlation_method (str): 'pearson' or 'spearman' correlation
        param_names (list, optional): Custom parameter names for display
        stat_names (list, optional): Custom statistic names for display
        save_path (str, optional): Path to save the plot
    """

    # Load the data
    params_df = pd.read_csv(params_path)
    stats_df = pd.read_csv(statistics_path)

    # Ensure same number of rows
    min_rows = min(len(params_df), len(stats_df))
    params_df = params_df.iloc[:min_rows]
    stats_df = stats_df.iloc[:min_rows]

    # reorder stats_df: peak_I, time_to_peak, growth_rate, area_I, final_R
    stats_df = stats_df[["peak_I", "time_to_peak", "growth_rate", "area_I", "final_R"]]

    # Get parameter and statistic column names
    param_cols = params_df.columns.tolist()
    stat_cols = stats_df.columns.tolist()

    # Use custom names if provided
    if param_names is not None:
        param_display_names = param_names
    else:
        param_display_names = [col.replace("_", " ") for col in param_cols]

    if stat_names is not None:
        stat_display_names = stat_names
    else:
        stat_display_names = [col.replace("_", " ").title() for col in stat_cols]

    # Calculate correlation matrix
    correlation_matrix = np.zeros((len(param_cols), len(stat_cols)))
    p_value_matrix = np.zeros((len(param_cols), len(stat_cols)))

    for i, param_col in enumerate(param_cols):
        for j, stat_col in enumerate(stat_cols):
            if correlation_method.lower() == "pearson":
                corr, p_val = pearsonr(params_df[param_col], stats_df[stat_col])
            elif correlation_method.lower() == "spearman":
                corr, p_val = spearmanr(params_df[param_col], stats_df[stat_col])
            else:
                raise ValueError("correlation_method must be 'pearson' or 'spearman'")

            correlation_matrix[i, j] = corr
            p_value_matrix[i, j] = p_val

    # Create DataFrame for easier handling
    corr_df = pd.DataFrame(
        correlation_matrix, index=param_display_names, columns=stat_display_names
    )

    apply_publication_style(font_size=16, axes_linewidth=1.5)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Create the heatmap
    mask = np.zeros_like(correlation_matrix, dtype=bool)

    # Use diverging colormap centered at 0
    heatmap = sns.heatmap(
        corr_df,
        annot=False,
        annot_kws={"fontweight": "bold", "fontsize": 9},  # Bold the annot font
        cmap="RdBu_r",  # Red-Blue diverging colormap
        center=0,
        square=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="black",
        ax=ax,
    )

    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, size=16)

    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # Make colorbar height consistent with axis
    cbar.ax.set_position(
        [
            cbar.ax.get_position().x0,
            ax.get_position().y0,
            cbar.ax.get_position().width,
            ax.get_position().height,
        ]
    )

    # Set specific tick labels
    cbar.set_ticks([-0.95, -0.5, 0, 0.5, 0.95])
    cbar.set_ticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"])

    # Adjust layout
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return corr_df, p_value_matrix


def plot_multiple_correlation_heatmaps(
    params_paths,
    statistics_paths,
    iteration_labels=None,
    correlation_method="pearson",
    param_names=None,
    stat_names=None,
    save_path=None,
):
    """Plot correlation heatmaps for multiple iterations side by side.

    Args:
        params_paths (list): List of paths to parameter CSV files
        statistics_paths (list): List of paths to statistics CSV files
        iteration_labels (list, optional): Labels for each iteration
        correlation_method (str): 'pearson' or 'spearman' correlation
        param_names (list, optional): Custom parameter names for display
        stat_names (list, optional): Custom statistic names for display
        save_path (str, optional): Path to save the plot
    """

    n_iterations = len(params_paths)

    # Set default iteration labels if not provided
    if iteration_labels is None:
        iteration_labels = [f"Iteration {i+1}" for i in range(n_iterations)]

    apply_publication_style(font_size=9, axes_linewidth=1.5)

    # Create subplots
    fig, axes = plt.subplots(1, n_iterations, figsize=(8 * n_iterations, 8))
    if n_iterations == 1:
        axes = [axes]

    all_corr_matrices = []

    for i, (params_path, stats_path, iteration_label) in enumerate(
        zip(params_paths, statistics_paths, iteration_labels)
    ):
        # Load the data
        params_df = pd.read_csv(params_path)
        stats_df = pd.read_csv(stats_path)

        # Ensure same number of rows
        min_rows = min(len(params_df), len(stats_df))
        params_df = params_df.iloc[:min_rows]
        stats_df = stats_df.iloc[:min_rows]

        # Get parameter and statistic column names
        param_cols = params_df.columns.tolist()
        stat_cols = stats_df.columns.tolist()

        # Use custom names if provided
        if param_names is not None:
            param_display_names = param_names
        else:
            param_display_names = [col.replace("_", " ").title() for col in param_cols]

        if stat_names is not None:
            stat_display_names = stat_names
        else:
            stat_display_names = [col.replace("_", " ").title() for col in stat_cols]

        # Calculate correlation matrix
        correlation_matrix = np.zeros((len(param_cols), len(stat_cols)))
        p_value_matrix = np.zeros((len(param_cols), len(stat_cols)))

        for pi, param_col in enumerate(param_cols):
            for si, stat_col in enumerate(stat_cols):
                if correlation_method.lower() == "pearson":
                    corr, p_val = pearsonr(params_df[param_col], stats_df[stat_col])
                elif correlation_method.lower() == "spearman":
                    corr, p_val = spearmanr(params_df[param_col], stats_df[stat_col])
                else:
                    raise ValueError("correlation_method must be 'pearson' or 'spearman'")

                correlation_matrix[pi, si] = corr
                p_value_matrix[pi, si] = p_val

        all_corr_matrices.append(correlation_matrix)

        # Create DataFrame for easier handling
        corr_df = pd.DataFrame(
            correlation_matrix, index=param_display_names, columns=stat_display_names
        )

        # Create the heatmap
        ax = axes[i]
        heatmap = sns.heatmap(
            corr_df,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar=i == n_iterations - 1,  # Only show colorbar on last plot
            cbar_kws=(
                {"label": f"{correlation_method.title()} Correlation Coefficient"}
                if i == n_iterations - 1
                else {}
            ),
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        # Add significance indicators
        for pi in range(len(param_cols)):
            for si in range(len(stat_cols)):
                p_val = p_value_matrix[pi, si]
                if p_val < 0.001:
                    significance = "***"
                elif p_val < 0.01:
                    significance = "**"
                elif p_val < 0.05:
                    significance = "*"
                else:
                    significance = ""

                if significance:
                    corr_val = correlation_matrix[pi, si]
                    ax.text(
                        si + 0.5,
                        pi + 0.7,
                        significance,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black" if abs(corr_val) < 0.5 else "white",
                    )

        # Customize each subplot
        ax.set_title(f"{iteration_label}", fontsize=12, fontweight="bold")

        # Only leftmost plot gets y-axis labels
        if i == 0:
            ax.set_ylabel("Parameters", fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # All plots get x-axis labels
        ax.set_xlabel("Summary Statistics", fontsize=11, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if i == 0:
            plt.setp(ax.get_yticklabels(), rotation=0)

    # Add main title
    fig.suptitle(
        f"{correlation_method.title()} Correlation: Parameters vs Summary Statistics Across Iterations",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Add significance legend to the figure
    legend_text = "Significance: * p<0.05, ** p<0.01, *** p<0.001"
    fig.text(
        0.02,
        0.02,
        legend_text,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return all_corr_matrices


def main():
    base_dir = "../../../SIR_OUTPUT/n3_t365_l30"
    true_params = np.array(pd.read_csv(f"{base_dir}/true_params.csv"))[0]
    target_values_path = f"{base_dir}/target_values.csv"
    n_particles = 512
    n_iterations = 5
    if 1:  # Visualize overlapping histories for each iteration
        print("Creating overlapping history visualizations...")
        for iteration in [0, 4]:  # range(n_iterations):
            print(f"Processing iteration {iteration}...")
            history_paths = [
                f"{base_dir}/n{n_particles}/iter_{iteration}/history/history_{particle}.csv"
                for particle in range(n_particles)
            ]
            param_path = f"{base_dir}/n{n_particles}/iter_{iteration}/all_param_df.csv"
            statistics_path = f"{base_dir}/n{n_particles}/iter_{iteration}/final_metrics.csv"
            best_result = find_best_params(statistics_path, target_values_path)
            best_particle_idx = best_result["best_particle_idx"]
            params = pd.read_csv(param_path)
            print(f"Best particle: {best_particle_idx}")
            print(f"Best parameters: {params.iloc[best_particle_idx]}")
            # save best parameters to csv, with column names
            best_param_df = params.iloc[best_particle_idx].to_frame().T
            # add column names to best_param_df
            best_param_df.to_csv(
                f"{base_dir}/n{n_particles}/iter_{iteration}/best_params.csv",
                index=False,
                header=True,
            )

            if 1:
                for compartment in ["I", "R", "S"]:
                    save_path = (
                        f"{base_dir}/n{n_particles}/iter_{iteration}/history_{compartment}_best.png"
                    )
                    visualize_results(
                        history_paths,
                        statistics_path,
                        target_values_path,
                        compartment=compartment,
                        save_path=save_path,
                    )
                    print(f"Saved: {save_path}")
    if 0:  # Plot best histories overlap with true history
        true_history_path = f"{base_dir}/model_history.csv"
        best_history_paths = [
            f"{base_dir}/n{n_particles}/iter_{n_iterations-1}/best_histories/model_history_{i}.csv"
            for i in range(30)
        ]
        save_path = (
            f"{base_dir}/n{n_particles}/iter_{n_iterations-1}/best_histories/best_histories.png"
        )
        visualize_results_best_histories(
            true_history_path,
            best_history_paths,
            target_values_path,
            compartments=["I", "R", "S"],
            save_path=save_path,
        )
    if 0:  # Plot joint distribution of parameters
        prior_params_path = f"{base_dir}/n{n_particles}/iter_0/all_param_df.csv"
        posterior_params_path = f"{base_dir}/n{n_particles}/iter_4/all_param_df.csv"
        prior_samples = np.array(pd.read_csv(prior_params_path))
        posterior_samples = np.array(pd.read_csv(posterior_params_path))
        param_names = ["PI", "PR", "IIF", "ISF"]

        print("\nShowing 2x3 compact layout version:")
        save_path = f"{base_dir}/n{n_particles}/joint_distribution_4params_compact.png"
        plot_joint_distribution_4params_compact(
            prior_samples, posterior_samples, true_params, param_names, save_path=save_path
        )
    if 0:  # Plot summary statistics of parameter errors
        statistics_paths = [
            f"{base_dir}/n{n_particles}/iter_{i}/final_metrics.csv" for i in range(n_iterations)
        ]
        save_path = f"{base_dir}/n{n_particles}/iter_4/parameter_errors.png"
        analyze_parameter_errors(statistics_paths, target_values_path, save_path=save_path)
    if 0:
        print("\n" + "=" * 60)
        print("ANALYZING BEST PARAMETERS...")
        print("=" * 60)
        best_results = analyze_best_params_per_iteration(base_dir, n_particles, n_iterations)
        return best_results

    if 0:
        params_paths = [
            f"{base_dir}/n{n_particles}/iter_{i}/all_param_df.csv" for i in range(n_iterations)
        ]
        statistics_paths = [
            f"{base_dir}/n{n_particles}/iter_{i}/final_metrics.csv" for i in range(n_iterations)
        ]
        iteration_labels = [f"Iteration {i}" for i in range(n_iterations)]
        save_path = f"{base_dir}/n{n_particles}/correlation_heatmaps.png"
        # plot_multiple_correlation_heatmaps(params_paths, statistics_paths, iteration_labels, save_path=save_path)
        plot_correlation_heatmap(params_paths[-1], statistics_paths[-1], save_path=save_path)


if __name__ == "__main__":
    results = main()
