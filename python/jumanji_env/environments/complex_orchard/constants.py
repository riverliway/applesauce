### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/constants.py ####
# The viewer constants were removed from the script
import jax.numpy as jnp
from typing import TypeAlias

# This is a type alias for Jax Arrays so we can specify the shape of the arrays
# Ignore the actual value of this variable, it is only used for type hinting
# Use the JaxArray in the type hinting like: JaxArray['num_rows', 'num_cols', etc...]
JaxArray: TypeAlias = dict

# Actions
NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
PICK = 5
DROP = 6

NUM_ACTIONS = 7

# Simulation
TICK_SPEED = 10

# Trees
TREE_DISTANCE_ROW = (220, 420)
TREE_DISTANCE_COL = (220, 420)
TREE_VARIATION = (-30, 30)
TREE_DIAMETER = (40, 60)
TREE_MISSING_PROBABILITY = 0.02
# How many apples can be on a tree
ORCHARD_FERTILITY = (20, 50)

# Apples
APPLE_DIAMETER = (5, 10)
APPLE_DENSITY = 5 # How dense the apples can be placed in the orchard. This is value defines the probability function x^APPLE_DENSITY.

# Bots
ROBOT_DIAMETER = 60
# The speed of the robot in cm per tick
ROBOT_MOVE_SPEED = 30 / TICK_SPEED
# The speed of the robot in radians per tick
ROBOT_TURN_SPEED = 60 * jnp.pi / 180 / TICK_SPEED
# The distance the bot can be from the apple or bot to interact with it
ROBOT_INTERACTION_DISTANCE = 20

# Baskets
BASKET_DIAMETER = 100

# Rewards
REWARD_COST_OF_STEP = -1
REWARD_OUT_OF_BOUNDS = -5
REWARD_BAD_PICK = -0.5
REWARD_BAD_DROP = -0.5
REWARD_COLLECT_APPLE = 20
