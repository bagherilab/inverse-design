from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
import logging
import numpy as np
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF
import graphviz


class ABCSMCRFBase:
    """
    Base implementation of ABC Sequential Monte Carlo with Random Forests.

    This class implements the core functionality shared between different ABC-SMC-RF variants.
    """

    def __init__(
        self,
        n_iterations: int = 5,
        rf_type: Literal["RF", "DRF"] = "DRF",
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
        criterion: Literal["CART", "MMD"] = "CART",
    ):
        """
        Initialize the ABC-SMC-RF model.

        Parameters:
        -----------
        n_iterations : int, default=5
            Number of SMC iterations.
        rf_type : {'RF', 'DRF'}, default='DRF'
            Type of random forest to use.
        n_trees : int, default=500
            Number of trees in each random forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
        random_state : int, optional
            Controls the randomization in the random forests.
        criterion : {'CART', 'MMD'}, default='CART'
            The function to measure the quality of a split (for DRF only).
        """
        self.n_iterations = n_iterations
        self.rf_type = rf_type
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.n_try = n_try
        self.random_state = random_state
        self.criterion = criterion

        # Initialize RNG
        self.rng = np.random.RandomState(random_state)

        # Storage for results
        self.parameter_samples = []
        self.parameter_columns = []
        self.weights = []
        self.statistics = []
        self.rf_models = []

    def _build_rf_model(self, t: int) -> None:
        """
        Build a random forest model for the current iteration.

        Parameters:
        -----------
        t : int
            Current iteration index.
        """
        parameters = self.parameter_samples[t]
        statistics = self.statistics[t]

        if self.rf_type == "RF":
            if parameters.shape[1] > 1:
                logging.warning(
                    "Multiple parameters detected but using ABC-RF (univariate). "
                    "Consider using rf_type='DRF' for multivariate inference."
                )

            # Create separate models for each parameter
            models = []
            for p in range(parameters.shape[1]):
                model = RF(
                    n_trees=self.n_trees,
                    min_samples_leaf=self.min_samples_leaf,
                    n_try=self.n_try,
                    random_state=self.rng.randint(0, 2**31),
                )
                model.fit(parameters[:, p], statistics)
                models.append(model)

            self.rf_models.append(models)

        else:  # DRF
            model = DRF(
                n_trees=self.n_trees,
                min_samples_leaf=self.min_samples_leaf,
                n_try=self.n_try,
                random_state=self.rng.randint(0, 2**31),
                criterion=self.criterion,
            )
            model.fit(parameters, statistics)
            self.rf_models.append([model])

    def _compute_weights(self, t: int) -> None:
        """
        Compute weights for the current samples.

        Parameters:
        -----------
        t : int
            Current iteration index.
        """
        if self.rf_type == "RF":
            # Combine weights from separate RF models
            combined_weights = np.ones(self.n_particles)

            for p, model in enumerate(self.rf_models[t]):
                weights_p = model.predict_weights(self.observed_statistics)
                combined_weights *= weights_p

            # Renormalize
            if np.sum(combined_weights) > 0:
                combined_weights /= np.sum(combined_weights)

            self.weights.append(combined_weights)

        else:  # DRF
            model = self.rf_models[t][0]
            weights = model.predict_weights(self.target_values)
            self.weights.append(weights)

    def _get_model_and_tree(self, iteration: int, tree_index: int):
        """Get the appropriate model and tree based on iteration and tree index."""
        if iteration < 0:
            iteration = len(self.rf_models) + iteration

        if iteration < 0 or iteration >= len(self.rf_models):
            print(
                f"Invalid iteration index {iteration}. Available iterations: 0-{len(self.rf_models)-1}"
            )
            return None, None

        models = self.rf_models[iteration]

        if self.rf_type == "DRF":
            if len(models) == 0:
                print("No models available for this iteration")
                return None, None

            model = models[0]
            if tree_index >= len(model.trees):
                print(f"Invalid tree index {tree_index}. Available trees: 0-{len(model.trees)-1}")
                return None, None

            tree = model.trees[tree_index]
        else:  # RF
            if tree_index >= len(models):
                print(f"Invalid model index {tree_index}. Available models: 0-{len(models)-1}")
                return None, None

            model = models[tree_index]
            if len(model.trees) == 0:
                print("No trees available for this model")
                return None, None

            tree = model.trees[0]

        return tree, model

    def _get_feature_names(self, tree, provided_names=None):
        """Generate or validate feature names for the tree."""
        n_stats = 0
        for node in tree["nodes"]:
            if not node.get("leaf", True) and "split_stat_idx" in node:
                n_stats = max(n_stats, node["split_stat_idx"] + 1)

        if provided_names is None:
            return [f"statistic_{i}" for i in range(n_stats)]
        elif len(provided_names) < n_stats:
            return provided_names + [f"statistic_{i}" for i in range(len(provided_names), n_stats)]
        return provided_names

    def _calculate_leaf_weights(self, tree, iteration, target_values):
        """Calculate weights for all leaf nodes in the tree."""
        leaf_weights = []
        for node in tree["nodes"]:
            if node.get("leaf", True):
                samples = node.get("samples", [])
                if len(samples) > 0:
                    if self.rf_type == "DRF":
                        weights = self.weights[iteration][samples]
                        avg_weight = np.mean(weights)
                    else:  # RF
                        weights = np.ones(len(samples))
                        for model in self.rf_models[iteration]:
                            weights *= model.predict_weights(target_values)[samples]
                        avg_weight = np.mean(weights)
                    leaf_weights.append(avg_weight)

        if leaf_weights:
            min_weight = min(leaf_weights)
            max_weight = max(leaf_weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
            return leaf_weights, min_weight, max_weight, weight_range
        return [], 0, 1, 1

    @staticmethod
    def _get_color_for_weight(weight, min_weight, weight_range):
        """Convert weight to a color between red (low) and green (high)."""
        normalized = (weight - min_weight) / weight_range
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        return f"#{red:02x}{green:02x}00"

    def _create_color_legend(self, dot, min_weight, max_weight, weight_range):
        """Create a color bar legend for the tree visualization."""
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Weight Legend")
            legend.attr(style="rounded")
            legend.attr(color="black")

            n_segments = 5
            for i in range(n_segments):
                normalized = i / (n_segments - 1)
                weight = min_weight + normalized * weight_range
                color = self._get_color_for_weight(weight, min_weight, weight_range)

                legend.node(
                    f"legend_{i}",
                    f"{weight:.2e}",
                    shape="box",
                    style="filled",
                    fillcolor=color,
                    fontcolor="black" if normalized > 0.5 else "white",
                )

                if i > 0:
                    legend.edge(f"legend_{i-1}", f"legend_{i}", style="invis")

            legend.attr(rankdir="LR")
            legend.attr(rank="sink")

    def _build_tree_recursive(
        self,
        main,
        tree,
        node_dict,
        processed_nodes,
        processed_edges,
        feature_names,
        iteration,
        max_depth,
        min_weight,
        weight_range,
        target_values,
    ):
        """Build the tree visualization recursively."""

        def add_node_to_graph(node_id, depth=0):
            if node_id in processed_nodes:
                return

            processed_nodes.add(node_id)
            node = node_dict[node_id]
            dot_node_id = f"node_{node_id}"

            sample_count = len(node.get("samples", []))

            if node.get("leaf", True):
                samples = node.get("samples", [])
                if len(samples) > 0:
                    if self.rf_type == "DRF":
                        weights = self.weights[iteration][samples]
                        avg_weight = np.mean(weights)
                    else:
                        weights = np.ones(len(samples))
                        for model in self.rf_models[iteration]:
                            weights *= model.predict_weights(target_values)[samples]
                        avg_weight = np.mean(weights)

                    color = self._get_color_for_weight(avg_weight, min_weight, weight_range)
                    label = f"Leaf Node\nsamples = {sample_count}\nweight = {avg_weight:.3e}"
                else:
                    color = "#lightgray"
                    label = f"Leaf Node\nsamples = {sample_count}"

                main.node(dot_node_id, label, fillcolor=color, style="filled,rounded")
            else:
                if "split_stat_idx" in node and node["split_stat_idx"] < len(feature_names):
                    stat_name = feature_names[node["split_stat_idx"]]
                else:
                    stat_name = f"feature_{node.get('split_stat_idx', 'unknown')}"

                threshold = node.get("threshold", 0)
                label = f"{stat_name} ≤ {threshold:.3f}\nsamples = {sample_count}"
                main.node(dot_node_id, label, fillcolor="lightblue", style="filled,rounded")

        def add_tree_recursive(node_id, depth=0, visited=None):
            if visited is None:
                visited = set()

            if node_id in visited or depth > max_depth:
                return

            visited.add(node_id)
            add_node_to_graph(node_id, depth)

            node = node_dict[node_id]
            dot_node_id = f"node_{node_id}"

            if not node.get("leaf", True) and "left_child" in node and "right_child" in node:
                left_child = node["left_child"]
                right_child = node["right_child"]

                left_edge = (node_id, left_child, "yes")
                right_edge = (node_id, right_child, "no")

                if left_edge not in processed_edges:
                    processed_edges.add(left_edge)
                    left_dot_id = f"node_{left_child}"
                    main.edge(dot_node_id, left_dot_id, "yes")
                    add_tree_recursive(left_child, depth + 1, visited.copy())

                if right_edge not in processed_edges:
                    processed_edges.add(right_edge)
                    right_dot_id = f"node_{right_child}"
                    main.edge(dot_node_id, right_dot_id, "no")
                    add_tree_recursive(right_child, depth + 1, visited.copy())

        add_tree_recursive(0)

    def plot_tree(
        self,
        iteration: int = -1,
        tree_index: int = 0,
        max_depth: int = 3,
        target_values: np.ndarray = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a visualization of a tree from the random forest."""
        tree, model = self._get_model_and_tree(iteration, tree_index)
        if tree is None:
            return

        feature_names = self._get_feature_names(tree, feature_names)

        dot = graphviz.Digraph(comment="Decision Tree")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled")

        node_dict = {node["id"]: node for node in tree["nodes"]}
        leaf_weights, min_weight, max_weight, weight_range = self._calculate_leaf_weights(
            tree, iteration, target_values
        )

        if leaf_weights:
            self._create_color_legend(dot, min_weight, max_weight, weight_range)

        with dot.subgraph(name="cluster_main") as main:
            main.attr(label="Decision Tree")
            processed_nodes = set()
            processed_edges = set()
            self._build_tree_recursive(
                main,
                tree,
                node_dict,
                processed_nodes,
                processed_edges,
                feature_names,
                iteration,
                max_depth,
                min_weight,
                weight_range,
                target_values,
            )

        output_path = save_path or f"tree_iter_{iteration}_index_{tree_index}"
        try:
            dot.render(output_path, format="png", cleanup=True)
            print(f"Tree visualization with color bar saved as {output_path}.png")
        except Exception as e:
            print(f"Error rendering tree: {e}")
            print("\nGraphviz DOT source:")
            print(dot.source)

    def get_iteration_results(self, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the parameters, statistics, and weights from a specific iteration."""
        if t < 0 or t >= len(self.parameter_samples):
            raise ValueError(
                f"Invalid iteration index {t}. Must be between 0 and {len(self.parameter_samples)-1}."
            )
        return self.parameter_samples[t], self.statistics[t], self.weights[t]

    def get_variable_importance(
        self, t: Optional[int] = None, n_statistics: Optional[int] = None
    ) -> np.ndarray:
        """Get the importance of each summary statistic."""
        if not self.rf_models:
            raise ValueError("No models available. Call fit() first.")

        if t is None:
            t = len(self.rf_models) - 1

        if t < 0 or t >= len(self.rf_models):
            raise ValueError(
                f"Invalid iteration index {t}. Must be between 0 and {len(self.rf_models)-1}."
            )

        if self.rf_type == "RF":
            combined_importances = np.zeros(n_statistics)
            for model in self.rf_models[t]:
                importances = model.variable_importance()
                combined_importances += importances
            if np.sum(combined_importances) > 0:
                combined_importances /= np.sum(combined_importances)
            return combined_importances
        else:  # DRF
            model = self.rf_models[t][0]
            return model.variable_importance()
