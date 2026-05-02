import numpy as np
import pandas as pd

from inverse_design.analyze.population_metrics import PopulationMetrics


def main():
    population_metrics = PopulationMetrics()
    glioblastoma_data = pd.read_csv("../../data/glioblastoma.csv")
    gbp03_data = glioblastoma_data[glioblastoma_data["type"] == "GBP03"]
    data = gbp03_data
    diameters = list(
        map(lambda x: 2 * (x / np.pi) ** 0.5 * 1000, data["area"].tolist())
    )  # convert mm^2 to um
    diameters = {f"seed_0": diameters}
    timestamps = data["time"].tolist()
    results = population_metrics.calculate_colony_growth(diameters, timestamps)
    print(results)


if __name__ == "__main__":
    main()
