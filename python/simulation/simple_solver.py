from astar import find_path

def make_simple_decision(environment) -> list[str]:
  """
  Makes a simple decision for each bot based on the current environment.

  :param: environment [OrchardSimulation2D] The environment to make decisions in.

  :return: list[str] The decisions for each bot.
  """

  decisions = ['idle'] * environment.num_bots
  
  pickable_locations = [(apple_loc[0] + d[0], apple_loc[1] + d[1]) for apple_loc in environment.apples for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
  for index, bot_location in enumerate(environment.bot_locations):
    # If there are no apples, do nothing
    if len(environment.apples) == 0:
      continue

    # If the bot is next to an apple, pick
    if bot_location in pickable_locations:
      decisions[index] = 'pick'
      continue

    # Move towards the nearest apple
    closest_apple = find_closest_location(bot_location, environment.apples)
    decisions[index] = astar(environment, bot_location, closest_apple)

  return decisions

def find_closest_location(start: tuple[int, int], locations: list[tuple[int, int]]) -> tuple[int, int]:
  """
  Finds the closest location to the start location from the list of locations using manhattan distance.

  :param: start [tuple[int, int]] The starting location.
  :param: locations [list[tuple[int, int]]] The list of locations to search.

  :return: tuple[int, int] The closest location.
  """
  return min(locations, key=lambda loc: abs(loc[0] - start[0]) + abs(loc[1] - start[1]))

def astar(environment, start: tuple[int, int], goal: tuple[int, int]) -> str:
  """
  Finds the path from the start to the goal using the A* algorithm with manhattan distance heuristic.

  :param: environment [OrchardSimulation2D] The environment to search in.
  :param: start [tuple[int, int]] The starting location.
  :param: goal [tuple[int, int]] The goal location.

  :return: [str] The first step to take to get to the goal.
  """
  def find_neighbors(loc):
    neighbors = [(loc[0] + d[0], loc[1] + d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    return [n for n in neighbors if environment.is_valid_location(n[0], n[1])]

  result = find_path(
    start,
    goal,
    neighbors_fnct=find_neighbors,
    distance_between_fnct=lambda loc1, loc2: abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
  )

  if result is None:
    print('No path found')
    print(f'Start: {start}')
    print(f'Goal: {goal}')
    return 'idle'
  
  next_loc = list(result)[1]
  dx = next_loc[0] - start[0]
  dy = next_loc[1] - start[1]

  if dx == 1:
    return 'right'
  elif dx == -1:
    return 'left'
  elif dy == 1:
    return 'down'
  else:
    return 'up'
