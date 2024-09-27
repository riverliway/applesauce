import numpy as np
import json
import math
from perlin_noise import PerlinNoise

class OrchardComplex2D:
  """
  The complex type of 2D orchard simulation that can be created.

  All locations are measured in centimeters (cm).
  Movement is tank-style of rotate left or right and move forward or backward.
  Apples need to be picked up and placed in a basket.
  There is fog of war, so the bots can only see a certain distance around them.
  """
  # How many simulation steps are in a second
  TICK_SPEED = 10

  # How dense the trees can be placed in the orchard. This is a tuple of +/- 3 standard deviations in cm.
  TREE_DISTANCE_ROW = [220, 420]
  TREE_DISTANCE_COL = [220, 420]
  TREE_VARIATION = [-30, 30]
  TREE_DIAMETER = [40, 60]
  TREE_MISSING_PROBABILITY = 0.02
  APPLE_DIAMETER = [5, 10]
  # How many apples can be on a tree
  ORCHARD_FERTILITY = [5, 20]

  # How dense the apples can be placed in the orchard. This is value defines the probability function x^APPLE_DENSITY.
  APPLE_DENSITY = 5

  # The diameter of the robot in cm
  ROBOT_DIAMETER = 60
  # The speed of the robot in cm per tick
  ROBOT_MOVE_SPEED = 10 / TICK_SPEED
  # The speed of the robot in degrees per tick
  ROBOT_TURN_SPEED = 30 / TICK_SPEED
  # The diameter of the basket in cm
  BASKET_DIAMETER = 100

  # The list of valid actions that can be taken by the bots.
  ACTIONS = ["forward", "backward", "left", "right", "pick", "drop", "idle"]

  # Penality for when the bot attempts to move out of bounds
  OUT_OF_BOUNDS_PENALTY = -10
  # Penalty for when the bot is near an obstacle
  NEAR_OBSTACLE_PENALTY = -0.5
  # Penalty for when the bot moves over an apple
  CRUSH_APPLE_PENALTY = -5
  # Reward for when the bot picks an apple
  PICK_APPLE_REWARD = 1
  # Reward for when the bot moves without any impact
  NO_IMPACT_REWARD = 0
  # Penalty for when the bot tries to pick an apple, but there is no apple to pick
  PICK_NO_APPLE_PENALTY = -0.5
  # Penalty for when the bot tries to pick an apple, but it is already holding an apple
  PICK_DUP_APPLE_PENALTY = -0.5
  # Reward for when the bot drops an apple in the basket
  DROP_APPLE_REWARD = 5
  # Penalty for when the bot drops an apple, but there is no basket to drop it in
  DROP_NO_BASKET_PENALTY = -0.75
  # Penalty for when the bot drops an apple, but there is no apple to drop
  DROP_NO_APPLE_PENALTY = -0.5

  def __init__(self, width: int, height: int, num_picker_bots: int = 1, num_pusher_bots: int = 0, num_baskets: int = 1, seed: int = None) -> None:
    """
    Instantiates a new OrchardSimulation2D object.

    :param: width [int] The width of the orchard.
    :param: height [int] The height of the orchard.
    :param: num_picker_bots [int] The number of apple picker bots to simulate.
    :param: num_pusher_bots [int] The number of basket pusher bots to simulate.
    :param: seed [int] The random seed to use for the simulation.

    :return: None
    """

    self.width = validate_int(width)
    self.height = validate_int(height)
    self.num_picker_bots = validate_int(num_picker_bots)
    self.num_pusher_bots = validate_int(num_pusher_bots)
    self.seed = validate_int(seed if seed is not None else np.random.randint(0, 1_000_000))
    np.random.seed(self.seed)

    # Time is measured in ticks
    self.time = 0

    self.__populate_orchard()

  def __populate_orchard(self) -> None:
    """
    Populates the orchard with trees, apples, and sets the initial locations of the bots.
    """
    # Perlin noise generator https://en.wikipedia.org/wiki/Perlin_noise
    # Used to decide how much fruit a tree can bear (fertility)
    noise_fertility = PerlinNoise(octaves=1, seed=self.seed)
    # Used to decide if a tree is present or not
    noise_tree = PerlinNoise(octaves=2, seed=self.seed + 1)

    # All locations are measured in cm and can be floats
    tree_row_dist = int(self.random_normal(*self.TREE_DISTANCE_ROW))
    tree_col_dist = int(self.random_normal(*self.TREE_DISTANCE_COL))

    self.trees = [
      {
        'x': int(x + self.random_normal(*self.TREE_VARIATION)),
        'y': int(y + self.random_normal(*self.TREE_VARIATION)),
        'diameter': int(self.random_normal(*self.TREE_DIAMETER)),
        # The fertility of the tree is a value between 0 and 1
        'fertility': (noise_fertility([x / self.width, y / self.height]) + 1) / 2
      }
      for x in range(tree_row_dist, self.width, tree_row_dist)
      for y in range(tree_col_dist, self.height, tree_col_dist)
      if (noise_tree([x / self.width, y / self.height]) + 1) / 2 > self.TREE_MISSING_PROBABILITY
    ]

    self.bots = [
      {
        'x': int(tree_row_dist // 2 + tree_row_dist * i),
        'y': int(tree_col_dist // 2),
        'holding': False,
        'job': 'picker' if i < self.num_picker_bots else 'pusher',
        'diameter': int(self.ROBOT_DIAMETER)
      }
      for i in range(self.num_picker_bots + self.num_pusher_bots)
    ]

    # Generate the apples using polar coordinates
    self.apples = [
      {
        'x': int(tree['x'] + radius * math.cos(theta)),
        'y': int(tree['y'] + radius * math.sin(theta)),
        'diameter': int(self.random_normal(*self.APPLE_DIAMETER))
      }
      for tree in self.trees
      for theta, radius in [
        (theta, 1 / np.random.rand() ** self.APPLE_DENSITY + tree['diameter'] / 2 + self.APPLE_DIAMETER[1])
        for theta in self.__generate_angles(tree)
      ]
    ]

    # Filter out the apples that are out of bounds
    self.apples = [
      apple
      for apple in self.apples
      if self.is_valid_location(apple['x'], apple['y'], apple['diameter'])
    ]

    self.starting_bots = [i for i in self.bots]
    self.starting_trees = [i for i in self.trees]
    self.starting_apples = [i for i in self.apples]

  def step(self, actions: list[str]) -> list[float]:
    """
    Advances the simulation by one time step.

    :param: actions [list[str]] A list of actions to take for each bot.
      These actions must be one of the valid actions in the ACTIONS list.

    :return: [list[float]] The reward for the current time step for each bot.
    """

    # Validate the actions
    for action in actions:
      if action not in self.ACTIONS:
        raise ValueError(f"Invalid action: {action}")
      
    if len(actions) != self.num_bots:
      raise ValueError(f"Expected {self.num_bots} actions, but got {len(actions)} instead")
    
    rewards = [0.0] * self.num_bots

    # Perform the bot locations
    for i, action in enumerate(actions):
      if action == "up":
        rewards[i] = self.__action_movement(i, 0, -1)
      elif action == "down":
        rewards[i] = self.__action_movement(i, 0, 1)
      elif action == "left":
        rewards[i] = self.__action_movement(i, -1, 0)
      elif action == "right":
        rewards[i] = self.__action_movement(i, 1, 0)
      elif action == "pick":
        rewards[i] = self.__action_pick(i)

    # Update the time
    self.time += 1

    return rewards
  
  def is_valid_location(self, x: int, y: int, diameter: int) -> bool:
    """
    Checks if this location is a valid location for the bot to move to.

    :param: x [int] The x-coordinate of the bot.
    :param: y [int] The y-coordinate of the bot.
    :param: diameter [int] The diameter of the bot.
    """
    return self.__policy_movement(x, y, diameter, None)[1]
  
  def __action_movement(self, bot_idx: int, dx: int, dy: int) -> float:
    """
    Moves the bot at the given index by the given dx and dy values.

    :param: bot_idx [int] The index of the bot to move.
    :param: dx [int] The change in x-coordinate.
    :param: dy [int] The change in y-coordinate.

    :return: [float] The reward for the current time step.
    """

    new_location = (self.bot_locations[bot_idx][0] + dx, self.bot_locations[bot_idx][1] + dy)
    reward, can_move_there = self.__policy_movement(new_location[0], new_location[1], self.bots[bot_idx]['diameter'], bot_idx)
    if can_move_there:
      self.bot_locations[bot_idx] = new_location

    return reward
  
  def __action_pick(self, bot_idx: int) -> float:
    """
    Allows the bot to pick an apple if it is at the location of an apple.

    :param: bot_idx [int] The index of the bot to pick the apple.

    :return: [float] The reward for the current time step.
    """

    bot_location = self.bot_locations[bot_idx]
    # The bot can pick an apple if it is in one of the four adjacent locations
    pick_locations = [(bot_location[0] + d[0], bot_location[1] + d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]

    for pick_location in pick_locations:
      if pick_location in self.apples:
        self.apples.remove(pick_location)
        return self.PICK_APPLE_REWARD

    return self.PICK_NO_APPLE_PENALTY
  
  def __policy_movement(self, x: int, y: int, diameter: int, bot_idx: int) -> tuple[float, bool]:
    """
    Checks if this location is a valid location for the bot to move to and returns the reward for the bot.

    :param: x [int] The x-coordinate of the bot.
    :param: y [int] The y-coordinate of the bot.
    :param: diameter [int] The diameter of the bot.
    :param: bot_idx [int] The index of the bot. Can be null if there is no bot to base this around.

    :return: [float, bool] The reward for the current time step and a flag determining if the bot can move there.
    """

    radius = diameter / 2

    # Penalize the bot if it is out of bounds
    if self.edge_distance(x, y, diameter) <= 0:
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs into a tree
    if any([self.circle_distance(tree['x'], tree['y'], tree['diameter'] / 2, x, y, radius) <= 0 for tree in self.trees]):
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs into another bot
    if any([self.circle_distance(bot['x'], bot['y'], bot['diameter'] / 2, x, y, radius) <= 0 for i, bot in enumerate(self.bots) if i != bot_idx]):
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs over an apple
    if any([self.circle_distance(apple['x'], apple['y'], apple['diameter'] / 2, x, y, radius) <= 0 for apple in self.apples]):
      return self.CRUSH_APPLE_PENALTY, True
    
    # Penalize the bot if it is near the edge of the orchard
    if self.edge_distance(x, y, diameter) < diameter:
      return self.NEAR_OBSTACLE_PENALTY, True
    
    # Penalize the bot if it is near an obstacle
    if any([self.circle_distance(tree['x'], tree['y'], tree['diameter'] / 2, x, y, radius) <= diameter for tree in self.trees]):
      return self.NEAR_OBSTACLE_PENALTY, True
    
    return self.NO_IMPACT_REWARD, True
  
  def to_dict(self) -> dict:
    """
    Converts the current state of the simulation to a dictionary.

    :return: dict
    """

    return {
      "width": self.width,
      "height": self.height,
      "seed": self.seed,
      "bots": self.bots,
      "starting_bots": self.starting_bots,
      "trees": self.trees,
      "starting_trees": self.starting_trees,
      "apples": self.apples,
      "starting_apples": self.starting_apples,
      "time": self.time
    }

  def to_json(self) -> str:
    """
    Converts the current state of the simulation to a JSON string.

    :return: str
    """

    return json.dumps(self.to_dict())
  
  @staticmethod
  def from_json(json_str: str) -> "OrchardSimulation2D":
    """
    Converts a JSON string to an OrchardSimulation2D object.

    :param: json_str [str] The JSON string to convert.

    :return: OrchardSimulation2D
    """

    data = json.loads(json_str)
    sim = OrchardSimulation2D(data["width"], data["height"])

    sim.num_bots = data["num_bots"]
    sim.seed = data["seed"]
    sim.bot_locations = data["bot_locations"]
    sim.starting_bot_locations = data["starting_bot_locations"]
    sim.trees = data["trees"]
    sim.starting_tree_locations = data["starting_tree_locations"]
    sim.apples = data["apples"]
    sim.starting_apple_locations = data["starting_apple_locations"]
    sim.time = data["time"]

    return sim
  
  @staticmethod
  def random_normal(small, large):
    """
    Returns a random number from a normal distribution defined by the small and large representing +/- 3 standard deviations.

    :param: small [int] -3 standard deviations.
    :param: large [int] +3 standard deviations.
    """

    min_val = min(small, large)
    max_val = max(small, large)

    return np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 6)
  
  @staticmethod
  def distance(x1, y1, x2, y2):
    """
    Returns the Euclidean distance between two points.
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  
  @staticmethod
  def circle_distance(x1, y1, r1, x2, y2, r2):
    """
    Returns the Euclidean distance between two circles
    """
    return OrchardComplex2D.distance(x1, y1, x2, y2) - r1 - r2
  
  def edge_distance(self, x, y, diameter):
    """
    Returns the manhattan distance to the edge of the orchard.
    """
    return min(x, y, self.width - x, self.height - y) - diameter / 2
  
  def __generate_angles(self, tree):
    """
    Generates the angles for the apples on the tree.
    """
    theta_offset = np.random.rand() * 2 * np.pi
    num_apples = max(self.random_normal(*self.ORCHARD_FERTILITY) * tree['fertility'], 0)

    angles = np.linspace(0, 2 * np.pi, int(num_apples) + 1)[:-1] + theta_offset
    angles += np.random.normal(0, 0.1, len(angles))

    return angles
  
def validate_int(value) -> int:
  """
  Validates that the given value is an integer.

  :param: value The value to validate.

  :return: int
  """

  if not isinstance(value, int):
    raise ValueError(f"Expected an integer, but got {type(value)}")

  return value


class OrchardSimulation2D:
  """
  The simplest type of orchard simulation that can be created.

  This simulation is a 2D grid where each cell can contain a tree, an apple, or a bot.
  The action and movements are very simple.
  """

  # How dense the trees can be placed in the orchard. This is a probability value.
  TREE_DENSITY = 0.05
  # How dense the apples can be placed in the orchard. This is a probability value.
  APPLE_DENSITY = 0.15

  # The list of valid actions that can be taken by the bots.
  ACTIONS = ["up", "down", "left", "right", "pick", "idle"]

  # Penality for when the bot attempts to move out of bounds
  OUT_OF_BOUNDS_PENALTY = -10
  # Penalty for when the bot is near an obstacle
  NEAR_OBSTACLE_PENALTY = -0.5
  # Penalty for when the bot moves over an apple
  CRUSH_APPLE_PENALTY = -5
  # Reward for when the bot picks an apple
  PICK_APPLE_REWARD = 1
  # Reward for when the bot moves without any impact
  NO_IMPACT_REWARD = 0
  # Penalty for when the bot tries to pick an apple, but there is no apple to pick
  PICK_NO_APPLE_REWARD = -0.5

  def __init__(self, width: int, height: int, num_bots: int = 1, seed: int = None) -> None:
    """
    Instantiates a new OrchardSimulation2D object.

    :param: width [int] The width of the orchard.
    :param: height [int] The height of the orchard.
    :param: num_bots [int] The number of bots to simulate.
    :param: seed [int] The random seed to use for the simulation.

    :return: None
    """

    self.width = validate_int(width)
    self.height = validate_int(height)
    self.num_bots = validate_int(num_bots)
    self.seed = validate_int(seed if seed is not None else np.random.randint(0, 1_000_000))
    np.random.seed(self.seed)
    self.time = 0

    self.__populate_orchard()

  def __populate_orchard(self) -> None:
    """
    Populates the orchard with trees, apples, and sets the initial locations of the bots.
    """
    # All locations are a tuple of (x, y) coordinates and are always integers
    self.bot_locations = [
      (math.floor((i + 1) / (self.num_bots + 1) * self.width), self.height // 2)
      for i in range(self.num_bots)
    ]

    self.trees = [
      (x, y)
      for x in range(self.width)
      for y in range(self.height)
      if np.random.rand() < self.TREE_DENSITY and (x, y) not in self.bot_locations
    ]

    self.apples = [
      (x, y)
      for x in range(self.width)
      for y in range(self.height)
      if np.random.rand() < self.APPLE_DENSITY and (x, y) not in self.bot_locations and (x, y) not in self.trees
    ]

    self.starting_bot_locations = [location for location in self.bot_locations]
    self.starting_tree_locations = [location for location in self.trees]
    self.starting_apple_locations = [location for location in self.apples]

  def step(self, actions: list[str]) -> list[float]:
    """
    Advances the simulation by one time step.

    :param: actions [list[str]] A list of actions to take for each bot.
      These actions must be one of the valid actions in the ACTIONS list.

    :return: [list[float]] The reward for the current time step for each bot.
    """

    # Validate the actions
    for action in actions:
      if action not in self.ACTIONS:
        raise ValueError(f"Invalid action: {action}")
      
    if len(actions) != self.num_bots:
      raise ValueError(f"Expected {self.num_bots} actions, but got {len(actions)} instead")
    
    rewards = [0.0] * self.num_bots

    # Perform the bot locations
    for i, action in enumerate(actions):
      if action == "up":
        rewards[i] = self.__action_movement(i, 0, -1)
      elif action == "down":
        rewards[i] = self.__action_movement(i, 0, 1)
      elif action == "left":
        rewards[i] = self.__action_movement(i, -1, 0)
      elif action == "right":
        rewards[i] = self.__action_movement(i, 1, 0)
      elif action == "pick":
        rewards[i] = self.__action_pick(i)

    # Update the time
    self.time += 1

    return rewards
  
  def is_valid_location(self, x: int, y: int) -> bool:
    """
    Checks if this location is a valid location for the bot to move to.
    """
    return self.__policy_movement(x, y)[1]
  
  def __action_movement(self, bot_idx: int, dx: int, dy: int) -> float:
    """
    Moves the bot at the given index by the given dx and dy values.

    :param: bot_idx [int] The index of the bot to move.
    :param: dx [int] The change in x-coordinate.
    :param: dy [int] The change in y-coordinate.

    :return: [float] The reward for the current time step.
    """

    new_location = (self.bot_locations[bot_idx][0] + dx, self.bot_locations[bot_idx][1] + dy)
    reward, can_move_there = self.__policy_movement(new_location[0], new_location[1])
    if can_move_there:
      self.bot_locations[bot_idx] = new_location

    return reward
  
  def __action_pick(self, bot_idx: int) -> float:
    """
    Allows the bot to pick an apple if it is at the location of an apple.

    :param: bot_idx [int] The index of the bot to pick the apple.

    :return: [float] The reward for the current time step.
    """

    bot_location = self.bot_locations[bot_idx]
    # The bot can pick an apple if it is in one of the four adjacent locations
    pick_locations = [(bot_location[0] + d[0], bot_location[1] + d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]

    for pick_location in pick_locations:
      if pick_location in self.apples:
        self.apples.remove(pick_location)
        return self.PICK_APPLE_REWARD

    return self.PICK_NO_APPLE_REWARD
  
  def __policy_movement(self, x: int, y: int) -> tuple[float, bool]:
    """
    Checks if this location is a valid location for the bot to move to and returns the reward for the bot.

    :param: x [int] The x-coordinate of the bot.
    :param: y [int] The y-coordinate of the bot.

    :return: [float, bool] The reward for the current time step and a flag determining if the bot can move there.
    """

    # Penalize the bot if it is out of bounds
    if x < 0 or x >= self.width or y < 0 or y >= self.height:
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs into a tree
    if (x, y) in self.trees:
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs into another bot
    if (x, y) in self.bot_locations:
      return self.OUT_OF_BOUNDS_PENALTY, False
    
    # Penalize the bot if it runs over an apple
    if (x, y) in self.apples:
      return self.CRUSH_APPLE_PENALTY, True
    
    # Penalize the bot if it is near the edge of the orchard
    if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
      return self.NEAR_OBSTACLE_PENALTY, True
    
    # Penalize the bot if it is near an obstacle
    for i in range(-1, 2):
      for j in range(-1, 2):
        if (x + i, y + j) in self.trees:
          return self.NEAR_OBSTACLE_PENALTY, True
    
    return self.NO_IMPACT_REWARD, True
  
  def to_dict(self) -> dict:
    """
    Converts the current state of the simulation to a dictionary.

    :return: dict
    """

    return {
      "width": self.width,
      "height": self.height,
      "num_bots": self.num_bots,
      "seed": self.seed,
      "bot_locations": self.bot_locations,
      "starting_bot_locations": self.starting_bot_locations,
      "trees": self.trees,
      "starting_tree_locations": self.starting_tree_locations,
      "apples": self.apples,
      "starting_apple_locations": self.starting_apple_locations,
      "time": self.time
    }

  def to_json(self) -> str:
    """
    Converts the current state of the simulation to a JSON string.

    :return: str
    """

    return json.dumps(self.to_dict())
  
  @staticmethod
  def from_json(json_str: str) -> "OrchardSimulation2D":
    """
    Converts a JSON string to an OrchardSimulation2D object.

    :param: json_str [str] The JSON string to convert.

    :return: OrchardSimulation2D
    """

    data = json.loads(json_str)
    sim = OrchardSimulation2D(data["width"], data["height"])

    sim.num_bots = data["num_bots"]
    sim.seed = data["seed"]
    sim.bot_locations = data["bot_locations"]
    sim.starting_bot_locations = data["starting_bot_locations"]
    sim.trees = data["trees"]
    sim.starting_tree_locations = data["starting_tree_locations"]
    sim.apples = data["apples"]
    sim.starting_apple_locations = data["starting_apple_locations"]
    sim.time = data["time"]

    return sim
  
def validate_int(value) -> int:
  """
  Validates that the given value is an integer.

  :param: value The value to validate.

  :return: int
  """

  if not isinstance(value, int):
    raise ValueError(f"Expected an integer, but got {type(value)}")

  return value
