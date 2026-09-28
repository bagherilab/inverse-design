"""
Run one SIR simulation and plot the S/I/R curves next to the final lattice.

    python run_example.py                 # show the figure
    python run_example.py --save out.png  # write it to a file instead
    python run_example.py --seed 0        # repeatable run

Needs numpy and matplotlib.
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from sir import SIR_ABM

# 0: empty, 1: susceptible, 2: infected, 3: recovered
STATE_COLORS = ["white", "#545aab", "#af1b0a", "#486b45"]
STATE_LABELS = ["Empty", "Susceptible", "Infected", "Recovered"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lattice-size", type=int, default=30)
    parser.add_argument("--infected-fraction", type=float, default=0.19)
    parser.add_argument("--susceptible-fraction", type=float, default=0.72)
    parser.add_argument("--Pm", type=float, default=1.0, help="migration rate")
    parser.add_argument("--PI", type=float, default=0.375, help="infection rate")
    parser.add_argument("--PR", type=float, default=0.005, help="recovery rate")
    parser.add_argument("--max-time", type=float, default=100.0)
    parser.add_argument("--record-interval", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", metavar="PATH", default=None)
    return parser.parse_args()


def plot(model, args):
    fig, (ax_curves, ax_lattice) = plt.subplots(1, 2, figsize=(12, 5))

    for state, color in zip("SIR", STATE_COLORS[1:]):
        ax_curves.plot(
            model.history["time"], model.history[state], color=color, label=state, linewidth=2
        )
    ax_curves.set_xlabel("Time")
    ax_curves.set_ylabel("Fraction of population")
    ax_curves.set_ylim(0, 1)
    ax_curves.set_title("SIR dynamics")
    ax_curves.legend(loc="best")
    ax_curves.grid(True, alpha=0.3)
    ax_curves.spines["top"].set_visible(False)
    ax_curves.spines["right"].set_visible(False)

    cmap = plt.cm.colors.ListedColormap(STATE_COLORS)
    im = ax_lattice.imshow(model.get_snapshot(), cmap=cmap, vmin=0, vmax=3)
    ax_lattice.set_xticks([])
    ax_lattice.set_yticks([])
    ax_lattice.set_title(f"Final state (t={model.time:.1f})")

    cbar = fig.colorbar(im, ax=ax_lattice, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(STATE_LABELS)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Figure written to {args.save}")
    else:
        plt.show()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = SIR_ABM(
        lattice_size=args.lattice_size,
        initial_infected_fraction=args.infected_fraction,
        initial_susceptible_fraction=args.susceptible_fraction,
        Pm=args.Pm,
        PI=args.PI,
        PR=args.PR,
    )
    model.run(max_time=args.max_time, record_interval=args.record_interval)

    stats = model.get_stats()
    print(f"Finished at t={stats['time']:.2f}")
    print(
        f"S={stats['susceptible']}  I={stats['infected']}  "
        f"R={stats['recovered']}  empty={stats['empty']}"
    )

    plot(model, args)


if __name__ == "__main__":
    main()
