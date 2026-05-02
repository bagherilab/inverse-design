import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pandas as pd
import os


class SIR_ABM:
    """
    SIR Agent-Based Model on a 2D square lattice with Von Neumann neighborhood.
    Key optimizations:
    1. Event-driven rate updates (only recalculate affected sites)
    2. Sparse data structures (track only occupied sites)
    3. Precomputed neighbor lists
    4. Early termination when no infections possible
    """

    def __init__(
        self,
        lattice_size=40,
        initial_infected_fraction=0.01,
        initial_susceptible_fraction=0.49,
        Pm=1.0,
        PI=0.05,
        PR=0.005,
    ):
        """
        Initialize the SIR ABM.

        Parameters:
        -----------
        lattice_size : int
            Size of the square lattice (X in the paper)
        initial_infected_fraction : float
            Fraction of lattice sites initially infected
        initial_susceptible_fraction : float
            Fraction of lattice sites initially susceptible
        Pm : float
            Migration rate
        PI : float
            Infection rate
        PR : float
            Recovery rate
        """
        self.X = lattice_size
        self.Pm = Pm
        self.PI = PI
        self.PR = PR
        self.initial_infected_fraction = initial_infected_fraction
        self.initial_susceptible_fraction = initial_susceptible_fraction

        # Initialize lattice
        # 0: empty, 1: susceptible, 2: infected, 3: recovered
        self.lattice = np.zeros((self.X, self.X), dtype=int)

        # Sparse data structures - track sites by state
        self.susceptible_sites = set()
        self.infected_sites = set()
        self.recovered_sites = set()
        self.empty_sites = set()

        # Precompute all neighbor relationships
        self.neighbors = {}
        for i in range(self.X):
            for j in range(self.X):
                self.neighbors[(i, j)] = self._get_von_neumann_neighbors(i, j)
                self.empty_sites.add((i, j))

        # Initialize agents
        self._initialize_agents()

        # Event storage with rates
        self.events = []  # List of (rate, event_type, pos1, pos2)
        self.total_rate = 0.0

        # Build initial event list
        self._rebuild_all_events()

        # Track time and history
        self.time = 0.0
        self.history = defaultdict(list)
        self._record_state()

    def _initialize_agents(self):
        """Initialize agent positions."""
        # Calculate number of agents
        total_sites = self.X * self.X
        n_infected = int(total_sites * self.initial_infected_fraction)
        n_susceptible = int(total_sites * self.initial_susceptible_fraction)

        if n_infected + n_susceptible > total_sites:
            print(
                f"n_infected: {n_infected}, n_susceptible: {n_susceptible}, total_sites: {total_sites}"
            )
            print(
                f"initial_infected_fraction: {self.initial_infected_fraction}, initial_susceptible_fraction: {self.initial_susceptible_fraction}"
            )
        assert (
            n_infected + n_susceptible <= total_sites
        ), "Total number of agents exceeds lattice size"

        # Randomly place agents
        all_positions = list(self.empty_sites)
        random.shuffle(all_positions)

        # Place infected agents
        for idx in range(n_infected):
            i, j = all_positions[idx]
            self.lattice[i, j] = 2
            self.infected_sites.add((i, j))
            self.empty_sites.remove((i, j))

        # Place susceptible agents
        for idx in range(n_infected, n_infected + n_susceptible):
            i, j = all_positions[idx]
            self.lattice[i, j] = 1
            self.susceptible_sites.add((i, j))
            self.empty_sites.remove((i, j))

    def _get_von_neumann_neighbors(self, i, j):
        """Get Von Neumann neighborhood with reflecting boundary conditions."""
        neighbors = []

        # Up
        if i > 0:
            neighbors.append((i - 1, j))
        else:
            neighbors.append((i, j))  # Reflecting boundary

        # Down
        if i < self.X - 1:
            neighbors.append((i + 1, j))
        else:
            neighbors.append((i, j))  # Reflecting boundary

        # Left
        if j > 0:
            neighbors.append((i, j - 1))
        else:
            neighbors.append((i, j))  # Reflecting boundary

        # Right
        if j < self.X - 1:
            neighbors.append((i, j + 1))
        else:
            neighbors.append((i, j))  # Reflecting boundary

        return neighbors

    def _get_site_events(self, pos):
        """Get all possible events for a given site."""
        i, j = pos
        current_state = self.lattice[i, j]
        events = []

        if current_state == 0:  # Empty site - no events
            return events

        neighbors = self.neighbors[pos]

        # Migration events (for S, I, R agents)
        if current_state in [1, 2, 3]:  # Agent present
            for ni, nj in neighbors:
                if (ni, nj) in self.empty_sites:  # Empty neighbor
                    events.append((self.Pm / 4, "migrate", pos, (ni, nj)))

        # Infection events
        if current_state == 2:  # Infected agent
            for ni, nj in neighbors:
                if (ni, nj) in self.susceptible_sites:  # Susceptible neighbor
                    events.append((self.PI / 4, "infect", pos, (ni, nj)))

        # Recovery events
        if current_state == 2:  # Infected agent
            events.append((self.PR, "recover", pos, None))

        return events

    def _rebuild_all_events(self):
        """Rebuild the complete event list (used at initialization)."""
        self.events = []

        # Only check occupied sites
        all_occupied = self.susceptible_sites | self.infected_sites | self.recovered_sites

        for pos in all_occupied:
            site_events = self._get_site_events(pos)
            self.events.extend(site_events)

        self.total_rate = sum(event[0] for event in self.events)

    def _update_events_for_sites(self, affected_sites):
        """Update events only for affected sites and their neighbors."""
        # Remove old events for affected sites
        sites_to_update = set(affected_sites)

        # Add neighbors of affected sites
        for site in affected_sites:
            sites_to_update.update(self.neighbors[site])

        # Remove events involving any of these sites
        old_events = self.events
        self.events = []
        removed_rate = 0.0

        for event in old_events:
            rate, event_type, pos1, pos2 = event
            # Check if this event involves any site we're updating
            involves_updated_site = pos1 in sites_to_update or (
                pos2 is not None and pos2 in sites_to_update
            )

            if involves_updated_site:
                removed_rate += rate
            else:
                self.events.append(event)

        # Add new events for updated sites
        new_rate = 0.0
        for site in sites_to_update:
            site_events = self._get_site_events(site)
            self.events.extend(site_events)
            new_rate += sum(event[0] for event in site_events)

        # Update total rate
        self.total_rate = self.total_rate - removed_rate + new_rate

    def gillespie_step(self):
        """Perform one Gillespie algorithm step."""
        # Early termination - no more infections possible
        if not self.infected_sites:
            return False

        if self.total_rate <= 0 or not self.events:
            return False

        # Time to next event
        dt = np.random.exponential(1.0 / self.total_rate)
        self.time += dt

        # Choose event
        r = random.random() * self.total_rate
        cumsum = 0
        selected_event = None

        for event in self.events:
            cumsum += event[0]
            if cumsum >= r:
                selected_event = event
                break

        if selected_event is None:
            return False

        rate, event_type, pos1, pos2 = selected_event
        affected_sites = set()

        # Execute event
        if event_type == "migrate":
            i1, j1 = pos1
            i2, j2 = pos2

            # Move agent
            agent_state = self.lattice[i1, j1]
            self.lattice[i2, j2] = agent_state
            self.lattice[i1, j1] = 0

            # Update site sets
            if agent_state == 1:
                self.susceptible_sites.remove(pos1)
                self.susceptible_sites.add(pos2)
            elif agent_state == 2:
                self.infected_sites.remove(pos1)
                self.infected_sites.add(pos2)
            elif agent_state == 3:
                self.recovered_sites.remove(pos1)
                self.recovered_sites.add(pos2)

            self.empty_sites.add(pos1)
            self.empty_sites.remove(pos2)

            affected_sites.update([pos1, pos2])

        elif event_type == "infect":
            i2, j2 = pos2
            self.lattice[i2, j2] = 2  # Susceptible becomes infected

            # Update site sets
            self.susceptible_sites.remove(pos2)
            self.infected_sites.add(pos2)

            affected_sites.add(pos2)

        elif event_type == "recover":
            i, j = pos1
            self.lattice[i, j] = 3  # Infected becomes recovered

            # Update site sets
            self.infected_sites.remove(pos1)
            self.recovered_sites.add(pos1)

            affected_sites.add(pos1)

        # Update events only for affected sites
        self._update_events_for_sites(affected_sites)

        return True

    def _record_state(self):
        """Record current state statistics."""
        total_occupied = (
            len(self.susceptible_sites) + len(self.infected_sites) + len(self.recovered_sites)
        )
        if total_occupied > 0:
            s_count = len(self.susceptible_sites)
            i_count = len(self.infected_sites)
            r_count = len(self.recovered_sites)

            self.history["time"].append(self.time)
            self.history["S"].append(s_count / total_occupied)
            self.history["I"].append(i_count / total_occupied)
            self.history["R"].append(r_count / total_occupied)
            self.history["total_infected"].append(i_count)

    def run(self, max_time=5.0, record_interval=0.1):
        """
        Run the simulation until max_time.

        Parameters:
        -----------
        max_time : float
            Maximum simulation time
        record_interval : float
            Time interval for recording state
        """
        next_record_time = record_interval

        while self.time < max_time:
            if not self.gillespie_step():
                print(f"Simulation ended early at time {self.time:.2f} (no more events possible)")
                break

            if self.time >= next_record_time:
                self._record_state()
                next_record_time += record_interval

        # Record final state
        self._record_state()

    def get_snapshot(self):
        """Get current lattice state for visualization."""
        return self.lattice.copy()

    def get_stats(self):
        """Get current simulation statistics."""
        return {
            "time": self.time,
            "susceptible": len(self.susceptible_sites),
            "infected": len(self.infected_sites),
            "recovered": len(self.recovered_sites),
            "empty": len(self.empty_sites),
            "total_events": len(self.events),
            "total_rate": self.total_rate,
        }


# Comparison function to test both models
def compare_models(max_time=50.0):
    """Compare performance of original vs SIR model."""
    import time

    # Parameters
    lattice_size = 30

    # print("Comparing Original vs Optimized SIR Model")
    # print("=" * 50)

    model = SIR_ABM(
        lattice_size=lattice_size,
        initial_infected_fraction=0.1929558435985482,
        initial_susceptible_fraction=0.7152248978220043,
        PI=0.3754264999312001,
        PR=0.0051245640383833,
        Pm=1.0,
    )

    model.run(max_time=max_time)

    return model


def visualize_results(model):
    """Visualize simulation results."""
    from inverse_design.plotting.theme import apply_publication_style

    apply_publication_style(font_size=12, axes_linewidth=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: S, I, R over time
    ax1.plot(model.history["time"], model.history["S"], "b-", label="Susceptible", linewidth=2)
    ax1.plot(model.history["time"], model.history["I"], "r-", label="Infected", linewidth=2)
    ax1.plot(model.history["time"], model.history["R"], "g-", label="Recovered", linewidth=2)

    # Style the first axis (consistent with previous figures)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.0)
    ax1.spines["bottom"].set_linewidth(1.0)
    ax1.tick_params(axis="both", which="major", labelsize=10, width=1.5)

    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Fraction of Population", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title("SIR Model Dynamics", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Current lattice state
    snapshot = model.get_snapshot()
    cmap = plt.cm.colors.ListedColormap(["white", "#545aab", "#af1b0a", "#486b45"])
    im = ax2.imshow(snapshot, cmap=cmap, vmin=0, vmax=3)

    # Add grid lines to show cell boundaries
    lattice_size = snapshot.shape[0]
    ax2.set_xticks(np.arange(-0.5, lattice_size, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, lattice_size, 1), minor=True)
    ax2.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.7)

    # Style the second axis (consistent with previous figures)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    ax2.tick_params(axis="both", which="minor", size=0)  # Hide minor tick marks
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_title(f"Final State (t={model.time:.1f})", fontsize=14)

    # Create colorbar with proper positioning and styling
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, aspect=20)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Empty", "Susceptible", "Infected", "Recovered"])
    cbar.ax.tick_params(labelsize=10, width=1.5)

    # Style the colorbar
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor("black")

    plt.tight_layout()
    plt.savefig("sir_model.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main function to run SIR simulation."""

    max_time = 365
    base_dir = "../../../SIR_OUTPUT/n3_t365_l30"
    n_particles = 1024
    n_iterations = 5

    for i in range(30):
        model = compare_models(max_time=max_time)
        if 1:
            model_history = pd.DataFrame(model.history)
            if not os.path.exists(
                f"{base_dir}/n{n_particles}/iter_{n_iterations-1}/best_histories"
            ):
                os.makedirs(f"{base_dir}/n{n_particles}/iter_{n_iterations-1}/best_histories")
            model_history.to_csv(
                f"{base_dir}/n{n_particles}/iter_{n_iterations-1}/best_histories/model_history_{i}.csv",
                index=False,
            )
    # Visualize results
    visualize_results(model)
    # Save model history to csv

    model_history = pd.DataFrame(model.history)
    print(model_history.head())
    model_history.to_csv("model_history.csv", index=False)
    # Print performance insights
    print(f"\nSimulation completed in {model.time:.2f} time units")
    print(
        f"Final state: S={len(model.susceptible_sites)}, I={len(model.infected_sites)}, R={len(model.recovered_sites)}"
    )


if __name__ == "__main__":
    main()
