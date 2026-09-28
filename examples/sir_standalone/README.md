# SIR agent-based model (standalone)

A lattice-based SIR agent-based model simulated with the Gillespie algorithm.
This directory is a self-contained copy of the simulator used in the
`inverse_design` project — you can drop `sir.py` into your own project and use
it without installing anything from this repository.

## Requirements

- `sir.py` needs **numpy** only.
- `run_example.py` additionally needs **matplotlib**.

```bash
pip install numpy matplotlib
```

## Quick start

```bash
python run_example.py                       # show the figure
python run_example.py --save sir.png        # write it to a file
python run_example.py --seed 0              # repeatable run
python run_example.py --PI 0.1 --PR 0.05    # change rates
python run_example.py --help                # all options
```

## Using it in your own code

```python
from sir import SIR_ABM

model = SIR_ABM(
    lattice_size=30,
    initial_infected_fraction=0.19,
    initial_susceptible_fraction=0.72,
    Pm=1.0,      # migration rate
    PI=0.375,    # infection rate
    PR=0.005,    # recovery rate
)
model.run(max_time=100.0, record_interval=0.1)

model.history      # dict of lists: "time", "S", "I", "R", "total_infected"
model.get_stats()  # dict: current time and per-compartment counts
model.get_snapshot()  # (X, X) int array of the lattice
```

`history["S"]`, `["I"]` and `["R"]` are fractions **of occupied sites**, not of
the whole lattice — empty sites are excluded from the denominator.
`history["total_infected"]` is a raw count. Turn the whole thing into a
DataFrame with `pandas.DataFrame(model.history)`.

## Parameters

| Parameter | Meaning |
|---|---|
| `lattice_size` | Side length `X` of the square lattice; the lattice has `X²` sites |
| `initial_infected_fraction` | Fraction of **all sites** seeded as infected |
| `initial_susceptible_fraction` | Fraction of **all sites** seeded as susceptible |
| `Pm` | Migration rate — an agent hops to an adjacent empty site |
| `PI` | Infection rate — an infected agent infects an adjacent susceptible |
| `PR` | Recovery rate — an infected agent becomes recovered |

The two initial fractions must sum to at most 1; the remaining sites start
empty. Placement is random and uniform.

## How the model works

Sites hold one of four states: empty (0), susceptible (1), infected (2),
recovered (3). Each site has a Von Neumann neighbourhood (up, down, left,
right).

Three event types fire, drawn by the Gillespie algorithm from the current
event list:

- **migrate**, rate `Pm/4` for each empty neighbour of an occupied site
- **infect**, rate `PI/4` for each susceptible neighbour of an infected site
- **recover**, rate `PR` for each infected site

After each event only the affected sites and their neighbours have their
events recomputed, rather than the whole lattice. `run()` stops at `max_time`,
or earlier if no infected agents remain (no further events are possible).

Two behaviours worth knowing if you extend the model:

- Boundaries are reflecting, implemented by mapping an out-of-bounds
  neighbour back to the site itself. A site on the edge therefore lists itself
  as a neighbour, which cannot host a migration or infection event — edge
  sites effectively have a lower total event rate than interior ones.
- Recovered agents still migrate; they just cannot be infected again.

## Reproducibility

The model draws from both the `random` and `numpy.random` global streams, so
seed both:

```python
import random, numpy as np
random.seed(0)
np.random.seed(0)
```

`run_example.py --seed N` does this for you.

## Provenance

`sir.py` contains the `SIR_ABM` class copied verbatim from
`src/inverse_design/models/sir/sir.py` in the `inverse_design` repository, so
runs here reproduce the behaviour reported in the paper. Only the
project-specific helper functions (which pulled in the project's plotting
theme and wrote to hardcoded output paths) were left out.

If you change the model, please say so when reporting results, so it stays
clear which version produced them.
