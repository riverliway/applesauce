### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/types.py ####

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass
    

# This replaces the Entity class.
# The level of the Entity was removed.
@dataclass
class SimpleOrchardEntity:
    """
    An entity that can be placed in the simple orchard environment (bot, apple, or tree).

    All fields are defined as a chex Array type with the shape as a comment beside it.
    A shape of () represents a scalar.

    id: unique number representing only this entity
    position: the position of this entity (x, y)
    """
    id: chex.Array # ()
    position: chex.Array #(2,)

@dataclass
class Agent(Entity):
    """
    An agent is an entity that can move and load food.

    id: unique number representing only this food.
    position: the position of this food.
    level: the level of this food.
    loading: whether the agent is currently loading food.
    """
    loading: chex.Array  # () - bool: is loading food

# This is replacing Food class.
@dataclass
class SimpleOrchardApple(SimpleOrchardEntity):
    """
    The desired collectable.
    """
    collected: chex.Array # () - bool: if the apple has been collected by a bot


# This is replacing State class
# agents were changed to bots and now directly calling the Entity class as opposed to an Agent subclass
# trees also added to the class with same logic
# apples replaces 'food_item'
# time replaces 'step_count'
@dataclass
class SimpleOrchardState:
    """
    Holds the state of the simple orchard using JAX fundementals
    """

    bots: SimpleOrchardEntity # List of bots (pytree structure)
    trees: SimpleOrchardEntity # List of trees (pytree structure)
    apples: SimpleOrchardApple # List of apples (pytree structure)
    step_count: chex.Array # ()
    key: chex.PRNGKey # (2,)

# this replaces the Observation class.
# agents_view now only holds two channels (x,y) but now applied to trees as well.
class SimpleOrchardObservation(NamedTuple):
    """
    The observation "seen" by the bots given to the neural network as the input layer.
    """
    # in the format (num_agents, [apple[i].position[0], apple[i].position[1], ..., bot[i].position[0], bot[i].position[1], ...])
    agents_view: chex.Array # (num_agents, 2 * (num_apples + num_trees + num_bots))
    action_mask: chex.Array # (num_agents, 6) since there are 6 actions in the simple env [UP, DOWN, LEFT, RIGHT, PICK, IDLE]
    step_count: chex.Array # ()