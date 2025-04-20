import os
import sys
import datetime

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

experiment_config = {
    "name": str(int(datetime.datetime.now().timestamp())),
    "seed": 42,
    "runs": 1,
    "population_size": 40,
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



def run_experiment(name='data', pop_size=20, speed=1, vision=10, separation=2, runs=50):
    all_data = []  # List to store data from each run

    for run_id in range(runs):
        model = BoidFlockers(
            population_size=pop_size,
            speed=speed,
            vision=vision,
            separation=separation)
        n = 0

        while model.running and n < 300:
            model.step()
            # print(f"Run {run_id + 1}, Step: {n}")
            n += 1

        df = model.datacollector.get_agent_vars_dataframe()
        df["run"] = run_id  # Add a column to indicate the run number
        all_data.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_data)
    combined_df.to_csv("../data/" + name + ".csv")    
    

run_experiments(experiment_config)