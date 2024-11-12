### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/constants.py ####
# The viewer constants were removed from the script

import jax.numpy as jnp

# Actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
LOAD = 5

MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
