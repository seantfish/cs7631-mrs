import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../../../.."))

from model import BoidFlockers as CBoidFlockers
from n_model import BoidFlockers as NNBoidFlockers

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

experiment_config = {
    "name": str(int(datetime.datetime.now().timestamp())),
    "seed": 42,
    "runs": 1,
    "timesteps": 1000,
    "population_size": 30,
    "speed": {
        "min": 1,
        "max": 11,
        "interval": 2,
    },
    "vision": {
        "min": 1,
        "max": 21,
        "interval": 4,
    },
    "separation": {
        "min": 1,
        "max": 11,
        "interval": 2
    }
}

def run_experiments(config):
    name = config['name']
    speeds = list(range(config['speed']['min'],
                        config['speed']['max'],
                        config['speed']['interval']))
    visions = list(range(config['vision']['min'],
                         config['vision']['max'],
                         config['vision']['interval']))
    separations = list(range(config['separation']['min'],
                             config['separation']['max'],
                             config['separation']['interval']))
    runs = config['runs']
    pop = config['population_size']
    print("Starting experiment ", name)
    print("Running experiments with parameters:")
    print("  Speeds:", speeds)
    print("  Visions:", visions)
    print("  Separations:", separations)

    os.mkdir("../data/" + name)

    i = 0
    for speed in speeds:
        for vision in visions:
            for separation in separations:
                print("RUN:", i)
                run_experiment(
                    name=(name + '/' + 'spd' + str(speed) + 'vis' + str(vision) + 'sep' + str(separation)),
                    pop_size=pop,
                    speed=speed,
                    vision=vision, 
                    separation=separation,
                    runs=runs)
                i += 1

    for speed in speeds:
        for vision in visions:
            for separation in separations:
                print("RUN:", i)
                run_experiment(
                    name=(name + '/' + 'spd' + str(speed) + 'vis' + str(vision) + 'sep' + str(separation)),
                    pop_size=pop,
                    speed=speed,
                    vision=vision, 
                    separation=separation,
                    runs=runs)
                i += 1

def run_experiment(name='data', 
                   pop_size=20, 
                   speed=1, vision=10, separation=2, runs=50,
                   version="classic"):
    all_agent_data = []  # List to store data from each run
    all_model_data = []

    for run_id in range(runs):
        if version == 'classic':
            model = CBoidFlockers(
                population_size=pop_size,
                speed=speed,
                vision=vision,
                separation=separation)
        elif version == 'neural':
            model = NNBoidFlockers(
                population_size=pop_size,
                speed=speed,
                vision=vision,
                separation=separation)
        n = 0

        while model.running and n < 1000:
            model.step()
            # print(f"Run {run_id + 1}, Step: {n}")
            n += 1

        df1 = model.datacollector.get_agent_vars_dataframe()
        df2 = model.datacollector.get_model_vars_dataframe()
        # print(df2.head())
        # df = pd.merge(df1, df2, on=['Step'])
        df1["run"] = run_id  # Add a column to indicate the run number
        all_agent_data.append(df1)
        all_model_data.append(df2)

    # Combine all dataframes
    combined_agent_df = pd.concat(all_agent_data)
    combined_agent_df.to_csv("../data/" + name + version + "_agent_" + ".csv")
    combined_model_df = pd.concat(all_model_data)
    combined_model_df.to_csv("../data/" + name + version + "_model_" + ".csv")    
    

run_experiments(experiment_config)