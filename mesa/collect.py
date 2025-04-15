import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from model import BoidFlockers

import pandas as pd

# def run_experiment():
#     model = BoidFlockers(population_size=20)
#     n = 0
#     while model.running and n < 1000:
#         model.step()
#         print("Step: ", n)
#         n = n + 1
#     df = model.datacollector.get_agent_vars_dataframe()
#     df.to_csv('flocking_250409_v4.csv')

def run_experiment():
    all_data = []  # List to store data from each run

    for run_id in range(50):
        model = BoidFlockers(population_size=20)
        n = 0
        while model.running and n < 300:
            model.step()
            print(f"Run {run_id + 1}, Step: {n}")
            n += 1

        df = model.datacollector.get_agent_vars_dataframe()
        df["run"] = run_id  # Add a column to indicate the run number
        all_data.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_data)
    combined_df.to_csv("flocking_250414_v2.csv")
    

run_experiment()