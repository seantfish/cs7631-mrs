"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""

import numpy as np
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
        self.neighbor_info = np.zeros((8)).tolist()

        self.nn_model = torch.load('../models/20250414_revamp', weights_only=False)
        self.nn_model.eval()


    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        n_neighbors, n_distances = self.get_nearest_neighbors(5)
        neighbors = np.array([n for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        distances = np.array([d for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        
        classic_direction = 0 + self.direction

        # NN Info
        neighbor_info = np.zeros((8)).tolist()

        # If no neighbors, maintain current direction
        if len(neighbors.tolist()) == 0:
            # Calculate diff_sum
            self.neighbor_diff_sum = [0, 0] # Early return here might have caused data issues
            neighbor_angles = np.array([])
        else:
            # Calculate diff_sum
            delta = self.space.calculate_difference_vector(self.position, agents=neighbors)
            self.neighbor_diff_sum = delta.sum(axis=0).tolist()

            self.neighbor_dists = delta.flatten()
            neighbor_angles = np.array([get_angle(n.direction) for n in neighbors])

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
            
        # Retrieve NN angle
        data = torch.tensor(neighbor_info, dtype=torch.float32)
        self.angle = self.nn_model(data).item()

        nn_direction = get_direction(self.angle)
        self.direction = nn_direction / np.linalg.norm(nn_direction)
        

        # Switch to classic
        # self.direction = classic_direction

        # Move boid
        self.position += self.direction * self.speed
        self.norm_dir = self.direction

        # Get angle information
        classic_norm_dir =  classic_direction
        classic_angle = get_angle(classic_norm_dir)

        # Compare
        angle_discrepancy = (self.angle - classic_angle) % math.pi

        print("==============================")
        print("CLASSIC DIRECTION: ", classic_direction)
        print("NN DIRECTION: ", nn_direction)
        print("SELF DIRECTION: ", self.direction)
        print("CLASSIC ANGLE: ", classic_angle)
        print("NN ANGLE: ", self.angle)
        print("ANGLE DISCREPANCY: ", angle_discrepancy)



def get_angle(direction):
    angle = np.arctan2(direction[1], direction[0])
    angle = (angle - (np.pi / 2)) % (2 * np.pi)
    return angle

def get_direction(angle):
    return [-1 * math.sin(angle), math.cos(angle)]