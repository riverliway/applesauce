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
    if bot['holding'] is not None and environment.can_drop(index) is not None:
      decisions[index] = 'drop'
      continue

    # Move towards the nearest target
    targets = [t for t in (environment.apples if bot['holding'] is None else environment.baskets) if not t['collected'] and not t['held']]
    closest_target = find_closest_location(bot, targets, environment.distance)

    target_type = 'apple' if bot['holding'] is None else 'basket'
    print_target = environment.apples[closest_target] if target_type == 'apple' else environment.baskets[closest_target]
    print(f'Moving towards {target_type} at {print_target}')

    decisions[index] = smart_move(environment, index, closest_target, target_type)

  print(decisions)
  return decisions

def find_closest_location(bot: dict, apples: list[dict], calc_dist) -> int:
  """
  Finds the closest apple to the bot's nose from a list of apples.

  :param: bot [dict] The bot to find the closest apple for.
  :param: apples [list[dict]] The list of apples to search through.

  :return: [int] The index of the closest apple to the bot's nose.
  """
  
  min_idx = 0
  min_dist = float('inf')

  nose_x = bot['x'] + math.cos(bot['orientation']) * bot['diameter'] / 2
  nose_y = bot['y'] + math.sin(bot['orientation']) * bot['diameter'] / 2

  for idx, apple in enumerate(apples):
    dist = calc_dist(nose_x, nose_y, apple['x'], apple['y'])
    if dist < min_dist:
      min_dist = dist
      min_idx = idx

  return min_idx

def smart_move(environment, bot_idx: int, goal_idx: int, goal_type: str) -> str:
  """
  Makes a move to get to the goal apple, only using astar when necessary.

  :param: environment [OrchardComplex2D] The environment to search in.
  :param: bot_idx [int] The starting location.
  :param: goal_idx [int] The index of the goal apple/basket.
  :param: goal_type [str] The type of the goal apple/basket.

  :return: [str] The first step to take to get to the goal.
  """

  goal = environment.apples[goal_idx] if goal_type == 'apple' else environment.baskets[goal_idx]

  nose_x = environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
  nose_y = environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2

  target_angle = math.atan2(goal['y'] - environment.bots[bot_idx]['y'], goal['x'] - environment.bots[bot_idx]['x'])

  # Check if the bot is blocked by an obstacle
  bot_blocked = any([
    line_intersects_circle(
      (obstacle['x'], obstacle['y']),
      obstacle['diameter'] / 2,
      # Check bot sides for collision
      (
        environment.bots[bot_idx]['x'] + math.cos(target_angle + side * math.pi / 2) * environment.bots[bot_idx]['diameter'] * 1.2 / 2,
        environment.bots[bot_idx]['y'] + math.sin(target_angle + side * math.pi / 2) * environment.bots[bot_idx]['diameter'] * 1.2 / 2
      ),
      (
        goal['x'] + math.cos(target_angle + side * math.pi / 2) * goal['diameter'] * 1.5 / 2,
        goal['y'] + math.sin(target_angle + side * math.pi / 2) * goal['diameter'] * 1.5 / 2
      )
    )
    for obstacle in [
      *environment.trees,
      *[b for i, b in enumerate(environment.bots) if i != bot_idx],
      *[b for i, b in enumerate(environment.baskets) if i != goal_idx or goal_type != 'basket']
    ]
    for side in (-1, 1)
  ])

  # if bot_blocked:
  for scale in (10, 5, 2):
    next_move = astar_basic(environment, bot_idx, goal, scale)
    if next_move != 'idle':
      return next_move
  
  return 'idle'
  
  # First check if turning the robot would help it face the target better
  angle_to_goal = math.atan2(goal['y'] - environment.bots[bot_idx]['y'], goal['x'] - environment.bots[bot_idx]['x'])
  current_angle = angle_difference(environment.bots[bot_idx]['orientation'], angle_to_goal)
  left_angle = angle_difference(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED, angle_to_goal)
  right_angle = angle_difference(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED, angle_to_goal)

  print(environment.bots[bot_idx]['orientation'], angle_to_goal, current_angle, left_angle, right_angle)

  if abs(left_angle) < abs(current_angle):
    return 'left'
  if abs(right_angle) < abs(current_angle):
    return 'right'
  
  # Then check if moving forward or backwards would help
  forward_dist = environment.distance(
    nose_x + math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
    nose_y + math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
    goal['x'],
    goal['y']
  )

  backward_dist = environment.distance(
    nose_x - math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
    nose_y - math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
    goal['x'],
    goal['y']
  )

  if forward_dist < backward_dist:
    return 'forward'
  else:
    return 'backward'
  
def astar_basic(environment, bot_idx: int, goal_target: dict, scale: int) -> str:
  """
  Finds the path from the start to the goal using the A* algorithm, but rendering a simple grid.

  :param: environment [OrchardComplex2D] The environment to search in.
  :param: bot_idx [int] The starting location.
  :param: goal_target [dict] The goal apple/basket.
  :param: scale [int] The scale to use for the grid.

  :return: [str] The first step to take to get to the goal.
  """
  SCALE = scale
  start = (
    (environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2) / SCALE,
    (environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2) / SCALE
  )

  goal = (round(goal_target['x'] / SCALE), round(goal_target['y'] / SCALE))

  is_valid_location = lambda x, y: environment.is_valid_location(
    x * SCALE - math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2,
    y * SCALE - math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2, 
    environment.bots[bot_idx]['diameter'],
    bot_idx
  )

  next_loc = astar_simple(is_valid_location, (round(start[0]), round(start[1])), goal, goal_target['diameter'] / 2 / SCALE)
  if (next_loc is None):
    return 'idle'

  neighbors = [
    # Forward
    (
      start[0] * SCALE + math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
      start[1] * SCALE + math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED
    ),
    # Backward
    (
      start[0] * SCALE - math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED,
      start[1] * SCALE - math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED
    ),
    # Left
    (
      environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED) * environment.bots[bot_idx]['diameter'] / 2,
      environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED) * environment.bots[bot_idx]['diameter'] / 2
    ),
    # Right
    (
      environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED) * environment.bots[bot_idx]['diameter'] / 2,
      environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED) * environment.bots[bot_idx]['diameter'] / 2
    )
  ]

  valid_moves = [
    n
    for n in neighbors
    if environment.is_valid_location(
      n[0] - math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2,
      n[1] - math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2,
      environment.bots[bot_idx]['diameter'], 
      bot_idx
  )]

  move_index = 0
  min_dist = float('inf')
  for idx, move in enumerate(valid_moves):
    dist = environment.distance(move[0], move[1], next_loc[0] * SCALE, next_loc[1] * SCALE)
    if dist < min_dist:
      move_index = idx + 1
      min_dist = dist

  return ['idle', 'forward', 'backward', 'left', 'right'][move_index]

def astar_simple(is_valid_location, start: tuple[int, int], goal: tuple[int, int], goal_radius: float) -> tuple[int, int]:
  """
  Finds the path from the start to the goal using the A* algorithm with manhattan distance heuristic.

  :param: environment [OrchardSimulation2D] The environment to search in.
  :param: start [tuple[int, int]] The starting location.
  :param: goal [tuple[int, int]] The goal location.
  :param: goal_radius [float] The radius of the goal.

  :return: [tuple[int, int]] The next location the bot should walk to
  """
  def find_neighbors(loc):
    # To avoid the basket collision interfering with the pathfinding,
    # If the bot is within 20% of the goal radius, it is at the goal
    if math.hypot(goal[0] - loc[0], goal[1] - loc[1]) < goal_radius * 1.2:
      return [goal]

    neighbors = [(loc[0] + d[0], loc[1] + d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    return [n for n in neighbors if is_valid_location(n[0], n[1])]

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
    return None
  
  return list(result)[1]

def astar(environment, bot_idx: int, goal_target: dict) -> str:
  """
  Finds the path from the start to the goal using the A* algorithm with manhattan distance heuristic.

  :param: environment [OrchardComplex2D] The environment to search in.
  :param: bot_idx [int] The starting location.
  :param: goal_target [dict] The goal apple/basket.

  :return: [str] The first step to take to get to the goal.
  """
  # We have to scale to avoid rounding errors but also not use floats
  SCALE = 10
  goal = (int(goal_target['x']) * SCALE, int(goal_target['y']) * SCALE)

  debug_called = 0

  def find_neighbors(loc, filter=True):
    # If the bot is next to an apple, its at the goal
    if environment.distance(loc[0], loc[1], goal[0], goal[1]) < environment.bots[bot_idx]['diameter'] / 2 * SCALE:
      return [goal]

    core_x = loc[0] - math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2 * SCALE
    core_y = loc[1] - math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2 * SCALE

    neighbors = [
      # Forward
      (
        int(loc[0] + math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED * SCALE),
        int(loc[1] + math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED * SCALE)
      ),
      # Backward
      (
        int(loc[0] - math.cos(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED * SCALE),
        int(loc[1] - math.sin(environment.bots[bot_idx]['orientation']) * environment.ROBOT_MOVE_SPEED * SCALE)
      ),
      # Left
      (
        int(core_x + math.cos(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED) * SCALE),
        int(core_y + math.sin(environment.bots[bot_idx]['orientation'] - environment.ROBOT_TURN_SPEED) * SCALE)
      ),
      # Right
      (
        int(core_x + math.cos(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED) * SCALE),
        int(core_y + math.sin(environment.bots[bot_idx]['orientation'] + environment.ROBOT_TURN_SPEED) * SCALE)
      )
    ]

    valid_moves = [
      n
      for n in neighbors
      if filter and environment.is_valid_location(
        n[0] / SCALE - math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2,
        n[1] / SCALE - math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2,
        environment.bots[bot_idx]['diameter'], 
        bot_idx
    )]

    return valid_moves
  
  def distance_between(loc1, loc2):
    """
    Calculates the distance between two locations.
    This distance is measures in how many simulation steps it would take to reach it.

    :param: loc1 [tuple[int, int]] The first location.
    :param: loc2 [tuple[int, int]] The second location.

    :return: [float] The number of steps between the two locations.
    """

    euclidean_distance = environment.distance(loc1[0], loc1[1], loc2[0], loc2[1])
    forward_steps = math.ceil(euclidean_distance / environment.ROBOT_MOVE_SPEED / SCALE)

    angle_distance = math.atan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
    turn_steps = math.ceil(angle_distance / environment.ROBOT_TURN_SPEED / SCALE)

    return forward_steps + turn_steps
  
  nose_x = environment.bots[bot_idx]['x'] + math.cos(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
  nose_y = environment.bots[bot_idx]['y'] + math.sin(environment.bots[bot_idx]['orientation']) * environment.bots[bot_idx]['diameter'] / 2
  
  start = (int(nose_x) * SCALE, int(nose_y) * SCALE)

  result = find_path(
    start,
    goal,
    neighbors_fnct=lambda loc: find_neighbors(loc, True),
    heuristic_cost_estimate_fnct=lambda loc1, loc2: environment.distance(loc1[0], loc1[1], loc2[0], loc2[1]),
    distance_between_fnct=distance_between
  )

  if result is None:
    print('No path found')
    print(f'Start: {start}')
    print(f'Goal: {goal}')
    return 'idle'
  
  next_loc = list(result)[1]
  possible_moves = find_neighbors((start[0], start[1]), filter=False)

  move_index = 0
  min_dist = float('inf')
  for idx, move in enumerate(possible_moves):
    dist = environment.distance(move[0], move[1], next_loc[0], next_loc[1])
    if dist < min_dist and environment.is_valid_location(move[0], move[1], environment.bots[bot_idx]['diameter']):
      move_index = idx
      break

  return ['forward', 'backward', 'left', 'right'][move_index]

def line_intersects_circle(circle_center, circle_radius, line_start, line_end):
    """
    Checks if a line segment intersects a circle.

    Args:
        circle_center: Tuple (x, y) representing the circle's center.
        circle_radius: Radius of the circle.
        line_start: Tuple (x, y) representing the start point of the line segment.
        line_end: Tuple (x, y) representing the end point of the line segment.

    Returns:
        True if the line segment intersects the circle, False otherwise.
    """

    # Unpack the points
    x1, y1 = line_start
    x2, y2 = line_end
    cx, cy = circle_center
    
    # Calculate the line segment direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the coefficients of the quadratic equation
    A = dx**2 + dy**2
    B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    C = (x1 - cx)**2 + (y1 - cy)**2 - circle_radius**2
    
    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C
    
    if discriminant < 0:
        # No intersection (the infinite line doesn't intersect the circle)
        return False
    
    # Find the t values where the line intersects the circle (parametric form)
    t1 = (-B + math.sqrt(discriminant)) / (2 * A)
    t2 = (-B - math.sqrt(discriminant)) / (2 * A)
    
    # Check if either t1 or t2 is within the segment bounds (0 <= t <= 1)
    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
        return True
    
    # If both t1 and t2 are out of bounds, the infinite line intersects but the segment does not
    return False

def angle_difference(angle1: float, angle2: float) -> float:
  """
  Calculates the difference between two angles.

  :param: angle1 [float] The first angle.
  :param: angle2 [float] The second angle.

  :return: [float] The difference between the two angles.
  """

  return (abs(angle1 - angle2) + math.pi) % (2 * math.pi) - math.pi
