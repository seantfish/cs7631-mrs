import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from model import BoidFlockers

import pandas as pd


def run_experiment():
    model = BoidFlockers()
    n = 0
    while model.running and n < 250:
        model.step()
        print("Step: ", n)
        n = n + 1
    df = model.datacollector.get_agent_vars_dataframe()
    df.to_csv('flocking_250301.csv')
    

run_experiment()