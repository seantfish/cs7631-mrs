import numpy as np

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
        
        # Initialize neighbor_info with proper structure for neural network input
        # Format: [own_angle, x_diff_sum, y_diff_sum, neighbor_angle1, neighbor_angle2, ...]
        self.neighbor_info = np.zeros(8).tolist()
        
        # Track additional data for model training
        self.prev_angle = self.angle
        self.delta_angle = 0
        self.neighbor_count = 0
        self.max_neighbors = 5

    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        # Store previous angle for calculating delta
        self.prev_angle = self.angle
        
        # Get nearest neighbors within vision radius
        n_neighbors, n_distances = self.get_nearest_neighbors(self.max_neighbors)
        neighbors = np.array([n for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        distances = np.array([d for n, d in zip(n_neighbors, n_distances) if n is not self and d < self.vision])
        
        # Store number of neighbors for potential analysis
        self.neighbor_count = len(neighbors)

        # Initialize structured neighbor info array with proper size
        neighbor_info = np.zeros(8)

        # If no neighbors, maintain current direction
        if len(neighbors.tolist()) == 0:
            self.position += self.direction * self.speed
            self.neighbor_diff_sum = [0, 0]
            
            # Set neighbor info with just the boid's own angle and zeros
            neighbor_info[0] = self.angle
            self.neighbor_info = neighbor_info.tolist()
            return

        # Calculate difference vectors to neighbors
        delta = self.space.calculate_difference_vector(self.position, agents=neighbors)
        
        # Store sum of neighbor position differences
        self.neighbor_diff_sum = delta.sum(axis=0).tolist()
        
        # Store flattened neighbor distance vectors
        self.neighbor_dists = delta.flatten()
        
        # Calculate angles of all neighbors
        self.neighbor_angles = np.array([get_angle(n.direction) for n in neighbors])

        # Calculate the three flocking behaviors
        cohere_vector = delta.sum(axis=0) * self.cohere_factor
        separation_vector = -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
        match_vector = np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
        
        # Update direction based on flocking rules
        self.direction += (cohere_vector + separation_vector + match_vector) / len(neighbors)
        
        # Normalize direction vector
        self.direction /= np.linalg.norm(self.direction)
        
        # Move boid
        self.position += self.direction * self.speed
        
        # Update normalized direction and angle
        self.norm_dir = self.direction
        self.angle = get_angle(self.norm_dir)
        
        # Calculate angle change from previous step (for supervised learning)
        self.delta_angle = angle_difference(self.angle, self.prev_angle)
        
        # Create structured neighbor_info array for neural network input
        # Index 0: Boid's own angle
        neighbor_info[0] = self.angle
        
        # Index 1-2: Sum of position differences to neighbors (x,y)
        neighbor_info[1] = self.neighbor_diff_sum[0]
        neighbor_info[2] = self.neighbor_diff_sum[1]
        
        # Index 3-7: Angles of up to 5 neighbors (padded with zeros if fewer)
        for i, angle in enumerate(self.neighbor_angles[:5]):
            neighbor_info[3 + i] = angle
            
        self.neighbor_info = neighbor_info.tolist()

# Helper function for angle calculations
def angle_difference(theta2, theta1):
    """Calculate the shortest angle difference between two angles in radians."""
    return np.arctan2(np.sin(theta2 - theta1), np.cos(theta2 - theta1))

def get_angle(direction):
    """Calculate the angle of a direction vector."""
    angle = np.arctan2(direction[1], direction[0])
    angle = (angle - (np.pi / 2)) % (2 * np.pi)
    return angle