import numpy as np
import json
import math

class OrchardSimulation2D:
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
