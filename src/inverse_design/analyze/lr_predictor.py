import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import networkx as nx
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import json
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks


class CorrelationPredictor:
    def __init__(self, r_threshold: float = 0.7):
        """
        Initialize the correlation-based predictor.

        Args:
            r_threshold: Correlation threshold for edge filtering
        """
        self.r_threshold = r_threshold
        self.correlation_matrix = None
        self.graphs = []
        self.predictions = {}
        self.models = {}

    def compute_correlations(self, data: pd.DataFrame) -> np.ndarray:
        """Compute pairwise correlations between parameters."""
        self.correlation_matrix = data.corr().values
        self.parameter_names = list(data.columns)
        return self.correlation_matrix

    def filter_edges(self) -> List[Tuple[int, int, float]]:
        """Filter edges based on correlation threshold."""
        edges = []
        n = len(self.correlation_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                r_val = abs(self.correlation_matrix[i, j])
                if r_val > self.r_threshold:
                    edges.append((i, j, r_val))

        return edges

    def build_graphs(self, edges: List[Tuple[int, int, float]]) -> List[nx.Graph]:
        """Build disconnected graphs from filtered edges."""
        G = nx.Graph()

        # Add all nodes (parameters)
        G.add_nodes_from(range(len(self.parameter_names)))

        # Add edges with correlation weights
        for i, j, r_val in edges:
            G.add_edge(i, j, weight=r_val)

        # Find connected components (disconnected graphs)
        self.graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        return self.graphs

    def find_starting_node(self, graph: nx.Graph) -> int:
        """Find the starting node with highest degree, then highest average correlation."""
        if len(graph.nodes()) == 0:
            return None

        # Calculate degrees
        degrees = dict(graph.degree())
        max_degree = max(degrees.values())

        # Find nodes with maximum degree
        max_degree_nodes = [node for node, deg in degrees.items() if deg == max_degree]

        if len(max_degree_nodes) == 1:
            return max_degree_nodes[0]

        # If multiple nodes have same degree, choose one with highest average correlation
        best_node = None
        best_avg_r = -1

        for node in max_degree_nodes:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 0:
                avg_r = np.mean(
                    [abs(self.correlation_matrix[node, neighbor]) for neighbor in neighbors]
                )
                if avg_r > best_avg_r:
                    best_avg_r = avg_r
                    best_node = node

        return best_node

    def predict_parameters(self, data: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Predict parameters using the greedy algorithm.

        Args:
            data: Input data with parameters as columns

        Returns:
            Dictionary mapping parameter index to predicted values
        """
        predictions = {}
        models = {}

        for graph in self.graphs:
            if len(graph.nodes()) <= 1:
                continue

            # Find starting node (highest degree, then highest average correlation)
            start_node = self.find_starting_node(graph)
            if start_node is None:
                continue

            # Track predicted nodes in this graph - start with the starting node
            predicted_nodes = {start_node}

            # Greedy prediction loop
            while True:
                best_r = -1
                best_model = None
                best_predictor = None
                best_target = None

                # Find the best next prediction
                for predictor in predicted_nodes:
                    for neighbor in graph.neighbors(predictor):
                        if neighbor not in predicted_nodes:
                            # Get correlation value
                            r_val = abs(self.correlation_matrix[predictor, neighbor])

                            if r_val > best_r:
                                # Try linear regression
                                model, r_squared = self.predict_with_linear_regression(
                                    data, predictor, neighbor
                                )

                                if (
                                    model is not None
                                ):  # and r_squared > 0.01:  # Minimum R² threshold
                                    best_r = r_val
                                    best_model = model
                                    best_predictor = predictor
                                    best_target = neighbor

                # If no good prediction found, stop
                if best_model is None:
                    break

                # Make prediction
                X_pred = data.iloc[:, best_predictor].values.reshape(-1, 1)
                mask = ~np.isnan(X_pred.flatten())

                predicted_values = np.full(len(data), np.nan)
                predicted_values[mask] = best_model.predict(X_pred[mask])

                predictions[best_target] = predicted_values
                models[best_target] = {
                    "model": best_model,
                    "predictor": best_predictor,
                    "correlation": best_r,
                    "r_squared": self.predict_with_linear_regression(
                        data, best_predictor, best_target
                    )[1],
                }

                predicted_nodes.add(best_target)

        self.predictions = predictions
        self.models = models
        return predictions

    def predict_with_linear_regression(
        self, data: pd.DataFrame, predictor_idx: int, target_idx: int
    ) -> Tuple[LinearRegression, float]:
        """Perform linear regression to predict target from predictor."""
        X = data.iloc[:, predictor_idx].values.reshape(-1, 1)
        y = data.iloc[:, target_idx].values

        # Remove rows with NaN values
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 2:
            return None, 0

        model = LinearRegression()
        model.fit(X_clean, y_clean)

        # Calculate R-squared
        y_pred = model.predict(X_clean)
        r_squared = r2_score(y_clean, y_pred)

        return model, r_squared

    def visualize_graphs(self, figsize: Tuple[int, int] = (10, 8)):
        """Visualize the correlation graphs."""
        if not self.graphs:
            print("No graphs to visualize. Run build_graphs() first.")
            return

        n_graphs = len(self.graphs)
        cols = min(3, n_graphs)
        rows = (n_graphs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_graphs == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, graph in enumerate(self.graphs):
            ax = axes[idx] if n_graphs > 1 else axes[0]

            # Create position layout
            pos = nx.spring_layout(graph, seed=42)

            # Draw nodes
            node_sizes = [300 + 100 * graph.degree(node) for node in graph.nodes()]
            nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="lightblue", ax=ax)

            # Draw edges with thickness based on correlation
            edges = graph.edges(data=True)
            edge_weights = [edge[2]["weight"] for edge in edges]
            nx.draw_networkx_edges(
                graph, pos, width=[w * 3 for w in edge_weights], alpha=0.6, ax=ax
            )

            # Add labels
            labels = {i: self.parameter_names[i] for i in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)

            ax.set_title(f"Graph {idx + 1} ({len(graph.nodes())} parameters)")
            ax.axis("off")

        # Hide unused subplots
        for idx in range(n_graphs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_prediction_summary(self) -> pd.DataFrame:
        """Get a summary of all predictions made."""
        if not self.models:
            return pd.DataFrame()

        summary_data = []
        for target_idx, model_info in self.models.items():
            summary_data.append(
                {
                    "Target Parameter": self.parameter_names[target_idx],
                    "Predictor Parameter": self.parameter_names[model_info["predictor"]],
                    "Prediction Rule": f"Use {self.parameter_names[model_info['predictor']]} → predict {self.parameter_names[target_idx]}",
                    "Correlation (r)": f"{model_info['correlation']:.3f}",
                    "R-squared": f"{model_info['r_squared']:.3f}",
                    "Slope": f"{model_info['model'].coef_[0]:.3f}",
                    "Intercept": f"{model_info['model'].intercept_:.3f}",
                }
            )

        return pd.DataFrame(summary_data)

    def save_prediction_summary_json(self, filename: str = "prediction_summary.json"):
        """Save prediction summary to JSON file with parameter categorization and models."""
        # if not self.models:
        #    print("No predictions to save.")
        #    return

        # Categorize parameters
        independent_params = []
        root_params = []
        derived_params = []

        # Track which parameters are in graphs
        params_in_graphs = set()

        # Find root parameters (starting nodes) and derived parameters
        for graph in self.graphs:
            if len(graph.nodes()) == 1:
                # Single node graphs are independent
                node = list(graph.nodes())[0]
                independent_params.append(
                    {"parameter_name": self.parameter_names[node], "parameter_index": node}
                )
            elif len(graph.nodes()) > 1:
                # Multi-node graphs have root and derived parameters
                root_node = self.find_starting_node(graph)
                if root_node is not None:
                    root_params.append(
                        {
                            "parameter_name": self.parameter_names[root_node],
                            "parameter_index": root_node,
                            "graph_id": len(root_params) + 1,
                            "graph_size": len(graph.nodes()),
                            "connections": len(list(graph.neighbors(root_node))),
                        }
                    )

                    # All other nodes in this graph are derived
                    for node in graph.nodes():
                        if node != root_node:
                            derived_params.append(
                                {
                                    "parameter_name": self.parameter_names[node],
                                    "parameter_index": node,
                                    "graph_id": len(root_params),  # Same graph as current root
                                    "root_parameter": self.parameter_names[root_node],
                                }
                            )

            # Track all parameters in graphs
            params_in_graphs.update(graph.nodes())

        # Any parameters not in any graph are also independent
        for i, param_name in enumerate(self.parameter_names):
            if i not in params_in_graphs:
                independent_params.append({"parameter_name": param_name, "parameter_index": i})

        # Create detailed summary for JSON
        summary_dict = {
            "parameter_categorization": {
                "independent_parameters": {
                    "description": "Parameters in disconnected graphs with only one node - no correlations with others",
                    "count": len(independent_params),
                    "parameters": independent_params,
                },
                "root_parameters": {
                    "description": "Starting nodes in multi-parameter graphs - highest degree and correlation",
                    "count": len(root_params),
                    "parameters": root_params,
                },
                "derived_parameters": {
                    "description": "Parameters predicted from root parameters using the models below",
                    "count": len(derived_params),
                    "parameters": derived_params,
                },
            },
            "prediction_models": {
                "description": "Linear regression models to predict derived parameters from root parameters",
                "total_models": len(self.models),
                "models": [],
            },
            "algorithm_settings": {
                "correlation_threshold": self.r_threshold,
                "minimum_r_squared": 0.1,
            },
        }

        # Add prediction models with full details needed for prediction
        for target_idx, model_info in self.models.items():
            model_data = {
                "target_parameter": {
                    "name": self.parameter_names[target_idx],
                    "index": target_idx,
                    "category": "derived",
                },
                "predictor_parameter": {
                    "name": self.parameter_names[model_info["predictor"]],
                    "index": model_info["predictor"],
                    "category": (
                        "root"
                        if model_info["predictor"] in [p["parameter_index"] for p in root_params]
                        else "derived"
                    ),
                },
                "linear_model": {
                    "equation": f"{self.parameter_names[target_idx]} = {model_info['model'].coef_[0]:.6f} * {self.parameter_names[model_info['predictor']]} + {model_info['model'].intercept_:.6f}",
                    "slope": model_info["model"].coef_[0],
                    "intercept": model_info["model"].intercept_,
                    "formula": "target = slope * predictor + intercept",
                },
                "prediction_quality": {
                    "correlation_r": model_info["correlation"],
                    "r_squared": model_info["r_squared"],
                    "correlation_strength": (
                        "strong"
                        if model_info["correlation"] > 0.8
                        else "moderate" if model_info["correlation"] > 0.6 else "weak"
                    ),
                },
            }
            summary_dict["prediction_models"]["models"].append(model_data)

        # Add sampling instructions
        summary_dict["sampling_instructions"] = {
            "description": "For efficient data collection, only sample these parameters:",
            "parameters_to_sample": {
                "independent": [p["parameter_name"] for p in independent_params],
                "root": [p["parameter_name"] for p in root_params],
            },
            "total_parameters_to_sample": len(independent_params) + len(root_params),
            "total_parameters_available": len(self.parameter_names),
            "reduction_efficiency": f"{(1 - (len(independent_params) + len(root_params)) / len(self.parameter_names)) * 100:.1f}%",
        }

        # Save to JSON file
        import json

        with open(filename, "w") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"Complete prediction summary saved to {filename}")
        print(
            f"Parameters to sample: {len(independent_params) + len(root_params)} out of {len(self.parameter_names)}"
        )
        print(
            f"Reduction efficiency: {summary_dict['sampling_instructions']['reduction_efficiency']}"
        )

        return summary_dict


# Example usage
def example_usage():
    """Demonstrate the correlation predictor with synthetic data."""
    np.random.seed(42)
    # Load the data
    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
    # base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N256_combined_grid_r2_0.7_p1"
    prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
    final_metrics_df = pd.read_csv(f"{base_dir}/iter_4/final_metrics.csv")
    target_metrics = json.load(open(f"{base_dir}/targets.json"))
    # keep only the columns that are in the target_metrics
    final_metrics_df = final_metrics_df[target_metrics.keys()]

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    prior_df.drop(columns=drop_cols, inplace=True)
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = prior_df.columns.tolist()
    param_subset = prior_df.columns.tolist()
    param_subset = [param for param in param_subset]  # if not param.endswith("_SIGMA")]
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        posterior_df, n_components=2
    )

    peak_idx = 0
    r_threshold = 0.01
    distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[peak_idx]) ** 2, axis=1))
    closest_indices = np.argsort(distances)[:50]
    peak_data = posterior_df.iloc[closest_indices]

    predictor = CorrelationPredictor(r_threshold=r_threshold)

    # Step 1: Compute correlations
    corr_matrix = predictor.compute_correlations(peak_data)
    print("Correlation Matrix:")
    print(pd.DataFrame(corr_matrix, columns=peak_data.columns, index=peak_data.columns).round(3))
    print()
    # Step 2: Filter edges and build graphs
    edges = predictor.filter_edges()
    print(f"Filtered edges (r > {predictor.r_threshold}): {len(edges)}")
    for i, j, r in edges:
        print(f"  {peak_data.columns[i]} - {peak_data.columns[j]}: r = {r:.3f}")
    print()

    # Step 3: Build graphs
    graphs = predictor.build_graphs(edges)
    print(f"Number of disconnected graphs: {len(graphs)}")
    for i, graph in enumerate(graphs):
        print(f"  Graph {i+1}: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        print(f"    Nodes: {[peak_data.columns[node] for node in graph.nodes()]}")
    print()

    # predictor.visualize_graphs()
    # Step 4-5: Predict parameters (assuming param_A and param_D are known)
    predictions = predictor.predict_parameters(peak_data)

    print("Prediction Results:")
    summary = predictor.get_prediction_summary()
    print(summary.to_string(index=False))
    print()
    json_data = predictor.save_prediction_summary_json(
        f"{base_dir}/posterior_plots/lr_predictions_r{r_threshold}_p{peak_idx+1}.json"
    )

    return predictor, peak_data, predictions


def predict_from_json_models(json_data: Dict, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict all derived parameters from sample data using JSON prediction models.

    Args:
        json_data: Dictionary containing parameter categorization and prediction models
        sample_df: DataFrame with only independent and root parameters

    Returns:
        DataFrame with all parameters (sampled + predicted)
    """

    # Initialize result dataframe with sample data
    result_df = sample_df.copy()

    # Get parameter information
    independent_params = [
        p["parameter_name"]
        for p in json_data["parameter_categorization"]["independent_parameters"]["parameters"]
    ]
    root_params = [
        p["parameter_name"]
        for p in json_data["parameter_categorization"]["root_parameters"]["parameters"]
    ]
    derived_params = [
        p["parameter_name"]
        for p in json_data["parameter_categorization"]["derived_parameters"]["parameters"]
    ]

    # Verify sample data has required parameters
    required_params = independent_params + root_params
    missing_params = set(required_params) - set(sample_df.columns)
    if missing_params:
        raise ValueError(f"Sample data missing required parameters: {missing_params}")

    print(f"Starting prediction for {len(derived_params)} derived parameters...")
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Required parameters present: {required_params}")

    # Initialize all derived parameters with NaN
    for param in derived_params:
        result_df[param] = np.nan

    # Build dependency graph from models
    models_dict = {}
    for model in json_data["prediction_models"]["models"]:
        target = model["target_parameter"]["name"]
        predictor = model["predictor_parameter"]["name"]
        slope = model["linear_model"]["slope"]
        intercept = model["linear_model"]["intercept"]
        r_squared = model["prediction_quality"]["r_squared"]

        models_dict[target] = {
            "predictor": predictor,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

    print(f"Loaded {len(models_dict)} prediction models")

    # Track which parameters have been predicted
    available_params = set(sample_df.columns)
    predicted_count = 0
    max_iterations = len(derived_params) * 2  # Safety limit
    iteration = 0

    # Iteratively predict parameters as their predictors become available
    while predicted_count < len(derived_params) and iteration < max_iterations:
        iteration += 1
        made_progress = False

        for target, model_info in models_dict.items():
            # Skip if already predicted
            if target in available_params:
                continue

            predictor = model_info["predictor"]

            # Check if predictor is available
            if predictor in available_params:
                # Apply linear model: target = slope * predictor + intercept
                slope = model_info["slope"]
                intercept = model_info["intercept"]
                r_squared = model_info["r_squared"]

                result_df[target] = slope * result_df[predictor] + intercept
                available_params.add(target)
                predicted_count += 1
                made_progress = True

                print(f"  Predicted {target} from {predictor} (R² = {r_squared:.3f})")

        # If no progress was made, we might have circular dependencies or missing predictors
        if not made_progress:
            remaining_targets = [
                target for target in models_dict.keys() if target not in available_params
            ]
            print(f"Warning: Could not predict remaining parameters: {remaining_targets}")

            # Show what predictors are missing for each remaining target
            for target in remaining_targets:
                predictor = models_dict[target]["predictor"]
                if predictor not in available_params:
                    print(f"  {target} needs {predictor} (not available)")
            break

    print(f"Prediction completed. Total parameters predicted: {predicted_count}")
    print(f"Final DataFrame shape: {result_df.shape}")

    # Reorder columns to match the original parameter order if possible
    try:
        # Get parameter order from indices in JSON
        all_params_with_indices = []

        for p in json_data["parameter_categorization"]["independent_parameters"]["parameters"]:
            all_params_with_indices.append((p["parameter_index"], p["parameter_name"]))

        for p in json_data["parameter_categorization"]["root_parameters"]["parameters"]:
            all_params_with_indices.append((p["parameter_index"], p["parameter_name"]))

        for p in json_data["parameter_categorization"]["derived_parameters"]["parameters"]:
            all_params_with_indices.append((p["parameter_index"], p["parameter_name"]))

        # Sort by index and get ordered parameter names
        all_params_with_indices.sort(key=lambda x: x[0])
        ordered_param_names = [name for idx, name in all_params_with_indices]

        # Reorder columns if all parameters are present
        if all(param in result_df.columns for param in ordered_param_names):
            result_df = result_df[ordered_param_names]
            print("Columns reordered to match original parameter indices")

    except Exception as e:
        print(f"Could not reorder columns: {e}")

    return result_df


def validate_predictions(
    json_data: Dict, predicted_df: pd.DataFrame, original_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Validate the predictions by showing model statistics and optionally comparing with original data.

    Args:
        json_data: Dictionary containing prediction models
        predicted_df: DataFrame with predicted values
        original_df: Optional DataFrame with original/true values for comparison

    Returns:
        DataFrame with validation statistics
    """
    validation_results = []

    for model in json_data["prediction_models"]["models"]:
        target = model["target_parameter"]["name"]
        predictor = model["predictor_parameter"]["name"]
        r_squared = model["prediction_quality"]["r_squared"]
        correlation_r = model["prediction_quality"]["correlation_r"]

        result = {
            "Target": target,
            "Predictor": predictor,
            "Model_R²": f"{r_squared:.3f}",
            "Correlation_r": f"{correlation_r:.3f}",
            "Prediction_Range": f"[{predicted_df[target].min():.4f}, {predicted_df[target].max():.4f}]",
        }

        # If original data is provided, calculate prediction accuracy
        if original_df is not None and target in original_df.columns:
            from sklearn.metrics import r2_score, mean_squared_error

            original_vals = original_df[target].values
            predicted_vals = predicted_df[target].values

            # Remove NaN values for comparison
            mask = ~(np.isnan(original_vals) | np.isnan(predicted_vals))
            if np.sum(mask) > 1:
                orig_clean = original_vals[mask]
                pred_clean = predicted_vals[mask]

                r2_actual = r2_score(orig_clean, pred_clean)
                rmse = np.sqrt(mean_squared_error(orig_clean, pred_clean))

                result["Actual_R²"] = f"{r2_actual:.3f}"
                result["RMSE"] = f"{rmse:.4f}"
            else:
                result["Actual_R²"] = "N/A"
                result["RMSE"] = "N/A"

        validation_results.append(result)

    return pd.DataFrame(validation_results)


# Additional utility functions
def evaluate_predictions(
    data: pd.DataFrame, predictions: Dict[int, np.ndarray], parameter_names: List[str]
) -> pd.DataFrame:
    """Evaluate prediction accuracy against actual values."""
    results = []

    for param_idx, pred_values in predictions.items():
        actual_values = data.iloc[:, param_idx].values

        # Remove NaN values for evaluation
        mask = ~(np.isnan(actual_values) | np.isnan(pred_values))
        if np.sum(mask) < 2:
            continue

        actual_clean = actual_values[mask]
        pred_clean = pred_values[mask]

        # Calculate metrics
        mse = np.mean((actual_clean - pred_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_clean - pred_clean))
        r_squared = r2_score(actual_clean, pred_clean)

        results.append(
            {
                "Parameter": parameter_names[param_idx],
                "RMSE": f"{rmse:.4f}",
                "MAE": f"{mae:.4f}",
                "R²": f"{r_squared:.4f}",
                "Sample Size": np.sum(mask),
            }
        )

    return pd.DataFrame(results)


def plot_predictions(
    data: pd.DataFrame,
    predictions: Dict[int, np.ndarray],
    parameter_names: List[str],
    max_plots: int = 6,
):
    """Plot actual vs predicted values for visual validation."""
    n_predictions = len(predictions)
    if n_predictions == 0:
        print("No predictions to plot.")
        return

    n_plots = min(n_predictions, max_plots)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (param_idx, pred_values) in enumerate(list(predictions.items())[:max_plots]):
        ax = axes[idx] if n_plots > 1 else axes[0]

        actual_values = data.iloc[:, param_idx].values

        # Remove NaN values
        mask = ~(np.isnan(actual_values) | np.isnan(pred_values))
        actual_clean = actual_values[mask]
        pred_clean = pred_values[mask]

        # Scatter plot
        ax.scatter(actual_clean, pred_clean, alpha=0.6, s=30)

        # Perfect prediction line
        min_val = min(np.min(actual_clean), np.min(pred_clean))
        max_val = max(np.max(actual_clean), np.max(pred_clean))
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect Prediction"
        )

        # Calculate R²
        r_squared = r2_score(actual_clean, pred_clean)

        ax.set_xlabel(f"Actual {parameter_names[param_idx]}")
        ax.set_ylabel(f"Predicted {parameter_names[param_idx]}")
        ax.set_title(f"{parameter_names[param_idx]}\nR² = {r_squared:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
