import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from n_model import BoidFlockers
from mesa.visualization import Slider, SolaraViz, make_space_component, make_plot_component
from matplotlib.path import Path
import matplotlib.colors as mcolors

import numpy as np

def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def boid_draw(agent):
    neighbors = len(agent.neighbors)

    
    norm_direction = agent.norm_dir

    arrow_points = np.array([[0, 1], [-1, -1], [0, 0], [1, -1], [0, 1]])

    angle = np.arctan2(norm_direction[1], norm_direction[0])
    angle = (angle - (np.pi / 2)) % (2 * np.pi)
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                           [np.sin(angle), np.cos(angle)]])

    arrow_marker = Path(arrow_points.dot(rot_matrix.T))

    color = "black"
    if agent.cluster > -1:
        color_keys = list(mcolors.TABLEAU_COLORS.keys())

        color = color_keys[agent.cluster % len(color_keys)]

    return {"color": color, "size": 20, "marker": arrow_marker}



model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "population_size": Slider(
        label="Number of boids",
        value=20,
        min=1,
        max=200,
        step=10,
    ),
    "width": 100,
    "height": 100,
    "speed": Slider(
        label="Speed of Boids",
        value=5,
        min=1,
        max=20,
        step=1,
    ),
    "vision": Slider(
        label="Vision of Bird (radius)",
        value=10,
        min=1,
        max=50,
        step=1,
    ),
    "separation": Slider(
        label="Minimum Separation",
        value=2,
        min=1,
        max=20,
        step=1,
    ),
}

model = BoidFlockers()

main_space = make_space_component(agent_portrayal=boid_draw, post_process=post_process, backend="matplotlib")

clusters_plot = make_plot_component("NumClusters")
std_plot = make_plot_component("StdHeading")

page = SolaraViz(
    model,
    components=[main_space, clusters_plot, std_plot],
    model_params=model_params,
    name="Boid Flocking Model",
)
page  # noqa
