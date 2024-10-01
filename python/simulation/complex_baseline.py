from astar import find_path
import math

def make_complex_decision(environment) -> list[str]:
  """
  Makes a complex decision for each bot based on the current environment.

  :param: environment [OrchardComplex2D] The environment to make decisions in.

  :return: list[str] The decisions for each bot.
  """

  decisions = ['idle'] * len(environment.bots)
  
  for index, bot in enumerate(environment.bots):
    # If there are no apples, do nothing
    if len([apple for apple in environment.apples if not apple['collected']]) == 0:
      continue

    # If the bot is next to an apple, pick
    if bot['holding'] is None and environment.try_pick(index) is not None:
      decisions[index] = 'pick'
      continue

    # If the bot is holding an apple next to a basket, drop
    if bot['holding'] is not None and environment.can_drop(index):
      decisions[index] = 'drop'
      continue

    # Move towards the nearest target
    targets = [t for t in (environment.apples if bot['holding'] is None else environment.baskets) if not t['collected'] and not t['held']]
    closest_target = find_closest_location(bot, targets, environment.distance)
    decisions[index] = astar(environment, index, closest_target)

  return decisions

def find_closest_location(bot: dict, apples: list[dict], calc_dist) -> int:
  """
  Finds the closest apple to the bot from a list of apples.

  :param: bot [dict] The bot to find the closest apple for.
  :param: apples [list[dict]] The list of apples to search through.

  :return: [int] The index of the closest apple to the bot.
  """
  
  min_idx = 0
  min_dist = float('inf')

  for idx, apple in enumerate(apples):
    dist = calc_dist(bot['x'], bot['y'], apple['x'], apple['y'])
    if dist < min_dist:
      min_dist = dist
      min_idx = idx

  return min_idx

def astar(environment, bot_idx: int, goal_apple_idx: int) -> str:
  """
  Finds the path from the start to the goal using the A* algorithm with manhattan distance heuristic.

  :param: environment [OrchardComplex2D] The environment to search in.
  :param: bot_idx [int] The starting location.
  :param: goal_apple_idx [int] The goal location.

  :return: [str] The first step to take to get to the goal.
  """
  def find_neighbors(loc, filter=True):
    core_x = loc[0] - math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
    core_y = loc[1] - math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2

    neighbors = [
      # Forward
      (
        int(loc[0] + math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED),
        int(loc[1] + math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED)
      ),
      # Backward
      (
        int(loc[0] - math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED),
        int(loc[1] - math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED)
      ),
      # Left
      (
        int(core_x + math.cos(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED)),
        int(core_y + math.sin(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED))
      ),
      # Right
      (
        int(core_x + math.cos(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED)),
        int(core_y + math.sin(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED))
      )
    ]

    return [n for n in neighbors if filter and environment.is_valid_location(n[0], n[1], environment.bots[bot_idx]['diameter'])]
  
  def distance_between(loc1, loc2):
    """
    Calculates the distance between two locations.
    This distance is measures in how many simulation steps it would take to reach it.

    :param: loc1 [tuple[int, int]] The first location.
    :param: loc2 [tuple[int, int]] The second location.

    :return: [float] The number of steps between the two locations.
    """

    euclidean_distance = environment.distance(loc1[0], loc1[1], loc2[0], loc2[1])
    forward_steps = math.ceil(euclidean_distance / environment.ROBOT_MOVE_SPEED)

    angle_distance = math.atan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
    turn_steps = math.ceil(angle_distance / environment.ROBOT_TURN_SPEED)

    return forward_steps + turn_steps
  
  nose_x = environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
  nose_y = environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
  
  start = (int(nose_x), int(nose_y))
  goal = (int(environment.apples[goal_apple_idx]['x']), int(environment.apples[goal_apple_idx]['y']))

  result = find_path(
    start,
    goal,
    neighbors_fnct=find_neighbors,
    distance_between_fnct=distance_between
  )

  if result is None:
    print('No path found')
    print(f'Start: {start}')
    print(f'Goal: {goal}')
    return 'idle'
  
  next_loc = list(result)[1]
  possible_moves = find_neighbors((nose_x, nose_y), filter=False)

  move_index = 0
  min_dist = float('inf')
  for idx, move in enumerate(possible_moves):
    dist = environment.distance(move[0], move[1], next_loc[0], next_loc[1])
    if dist < min_dist and environment.is_valid_location(move[0], move[1], environment.bots[bot_idx]['diameter']):
      move_index = idx
      break

  return ['forward', 'backward', 'left', 'right'][move_index]
