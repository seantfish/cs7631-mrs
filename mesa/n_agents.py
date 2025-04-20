"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import math

from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Boid(ContinuousSpaceAgent):
    """A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents
        - Separation: avoiding getting too close to any other agent
        - Alignment: trying to fly in the same direction as neighbors

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and direction (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=1,
        direction=(1, 1),
        vision=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
    ):
        """Create a new Boid flocker agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(space, model)
        self.position = position
        self.speed = speed
        self.direction = direction
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.neighbor_diff_sum = [0, 0]
        self.norm_dir = self.direction
        self.angle = get_angle(self.norm_dir)
        self.cluster = -1
        self.neighbor_info = np.zeros((17)).tolist()

        self.nn_model = torch.load('../models/20250419_v5', weights_only=False)
        self.nn_model.eval()
        
        self.angle_discrepancy = 0.0

        self.warm_start_steps = 50


    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        n_neighbors, n_distances = self.get_nearest_neighbors(5)
        neighbors = np.array([n for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        distances = np.array([d for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        
        classic_direction = copy.deepcopy(self.norm_dir)
        prev_norm_dir = copy.deepcopy(self.norm_dir)

        # If no neighbors, maintain current direction
        if len(neighbors.tolist()) == 0:
            # Calculate diff_sum
            self.neighbor_diff_sum = [0, 0] # Early return here might have caused data issues
            neighbor_angles = np.array([])
            # neighbor_presences = np.array([])
        else:
            # Calculate diff_sum
            delta = self.space.calculate_difference_vector(self.position, agents=neighbors)
            self.neighbor_diff_sum = delta.sum(axis=0).tolist()

            self.neighbor_dists = delta.flatten()
            neighbor_angles = np.array([i for n in neighbors for i in (n.direction / np.linalg.norm(n.direction))])
            # neighbor_presences = np.ones([neighbor_angles.shape[0]])

            # Cohere vector
            cohere_vector = delta.sum(axis=0) * self.cohere_factor

            # Separation vector
            separation_vector = (
                -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
            )

            # Match vector
            match_vector = (
                np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
            )

            # Update direction based on the three behaviors
            classic_direction += (cohere_vector + separation_vector + match_vector) / len(
                neighbors
            )

            # Normalize direction vector
            classic_direction /= np.linalg.norm(classic_direction)

        neighbor_info = np.zeros((17)).tolist()

        neighbor_info += np.pad([self.speed], (0, 16), 'constant', constant_values=0)
        neighbor_info += np.pad([self.vision], (1, 15), 'constant', constant_values=0)
        neighbor_info += np.pad([self.separation], (2, 14), 'constant', constant_values=0)

        neighbor_info += np.pad(prev_norm_dir, (3, 12), 'constant', constant_values=0)

        neighbor_info += np.pad(self.neighbor_diff_sum, (5, 10), 'constant', constant_values=0)

        neighbor_angles = np.pad(neighbor_angles, (0, 10 - neighbor_angles.shape[0]), 'constant', constant_values=0)
        neighbor_info += np.pad(neighbor_angles, (7, 0), 'constant', constant_values=0)

        neighbor_info = neighbor_info.tolist()

        neighbor_info = normalize_neighbor_info(neighbor_info)

        # assert neighbor_info[ 6.29, 'angle must not exceed 6.28, but found ' + str(neighbor_info[0])

        # Retrieve NN angle
        data = torch.tensor(neighbor_info, dtype=torch.float32)
        output = self.nn_model(data)
        # self.angle = max(min(self.angle, 1.0), 0.0)
        self.norm_dir = [output[0].item(), output[1].item()] / np.linalg.norm(self.norm_dir)
        # print(self.norm_dir)

        self.angle = get_angle(self.norm_dir) % (2 * np.pi)

        # Move boid
        if self.warm_start_steps <= 0:
            self.position += self.norm_dir * self.speed
        else: 
            self.position += classic_direction * self.speed
            self.norm_dir = classic_direction
            

        # Get angle information
        classic_norm_dir =  classic_direction
        classic_angle = get_angle(classic_norm_dir)


        # Compare 
        angle_discrepancy = abs(self.angle - classic_angle) % (2 * math.pi) 
        centered_nn_angle = self.angle 
        centered_cl_angle = classic_angle 
        cos_component = math.cos(centered_nn_angle - centered_cl_angle)
        sin_component = math.sin(centered_nn_angle - centered_cl_angle)
        angle_discrepancy = abs(math.atan2(sin_component, cos_component))
        

        if angle_discrepancy > 0:
            print("==============================")
            print("NEIGHBOR INFO: ", np.round(neighbor_info, decimals=2))
            print("PREV DIR: ", neighbor_info[3:5])
            print("PREV ANGLE: ", get_angle(neighbor_info[3:5]))
            # print("CLASSIC DIRECTION: ", classic_direction)
            # print("NN DIRECTION: ", get_angle(self.norm_dir))
            print("CLASSIC ANGLE: ", classic_angle)
            print("NN ANGLE: ", self.angle)
            print("ANGLE DISCREPANCY: ", angle_discrepancy)

        if self.warm_start_steps <= 0:
            self.angle_discrepancy = angle_discrepancy
        else: 
            self.angle_discrepancy = 0
            print("WARNING: WARM START ENABLED")
            self.warm_start_steps -= 1

        # Revert angle back to normalized 
        self.angle /= 2 * np.pi

def get_angle(direction):
    angle = np.arctan2(direction[1], direction[0])
    angle = (angle - (np.pi / 2)) % (2 * np.pi)
    return angle

def get_direction(angle):
    return [-1 * math.sin(angle), math.cos(angle)]

def normalize_neighbor_info(arr):
    arr = np.array(arr)
    norm_arr = arr.copy()

    norm_arr[0] = norm_arr[0] / 10
    norm_arr[1] = norm_arr[1] / 20
    norm_arr[2] = norm_arr[2] / 10

    # Angle is pre-normalized

    min_ndiff = -21 * 5
    max_ndiff = 21 * 5

    norm_arr[5] = (arr[5] - min_ndiff) / (max_ndiff - min_ndiff)
    norm_arr[6] = (arr[6] - min_ndiff) / (max_ndiff - min_ndiff)

    # On/off stays same
    return norm_arr.tolist()
