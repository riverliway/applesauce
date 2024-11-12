### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/constants.py ####
# The viewer constants were removed from the script
import jax.numpy as jnp

# Actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PICK = 5
DROP = 6

NUM_ACTIONS = 7

# Simulation
TICK_SPEED = 10

# Trees
TREE_DISTANCE_ROW = [220, 420]
TREE_DISTANCE_COL = [220, 420]
TREE_VARIATION = [-30, 30]
TREE_DIAMETER = [40, 60]
TREE_MISSING_PROBABILITY = 0.02
# How many apples can be on a tree
ORCHARD_FERTILITY = [20, 50]

# Apples
APPLE_DIAMETER = [5, 10]
APPLE_DENSITY = 5 # How dense the apples can be placed in the orchard. This is value defines the probability function x^APPLE_DENSITY.

# Bots
ROBOT_DIAMETER = 60
# The speed of the robot in cm per tick
ROBOT_MOVE_SPEED = 30 / TICK_SPEED
# The speed of the robot in radians per tick
ROBOT_TURN_SPEED = 60 * jnp.pi / 180 / TICK_SPEED

# Baskets
BASKET_DIAMETER = 100
