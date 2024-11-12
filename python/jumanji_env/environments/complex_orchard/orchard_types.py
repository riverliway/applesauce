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
class ComplexOrchardEntity:
    """
    An entity that can be placed in the complex orchard environment (bot, basket, apple, or tree).

    All fields are defined as a chex Array type with the shape as a comment beside it.
    A shape of () represents a scalar.

    id: unique number representing only this entity
    position: the position of this entity (x, y) in centimeters
    diameter: the diameter of this entity in centimeters
    """
    id: chex.Array # ()
    position: chex.Array #(2,)
    diameter: chex.Array # ()

@dataclass
class ComplexOrchardBot(ComplexOrchardEntity):
    """
    An agent is an entity that can move and load apples.

    id: unique number representing only this agent.
    position: the position of this entity (x, y) in centimeters
    diameter: the diameter of this entity in centimeters
    holding: the index of the apple that the agent is holding. If the agent is not holding an apple, this value is -1.
    job: int: 0 if the agent is an apple collector, 1 if the agent is a basket transporter
    orientation: float: the angle in radians that the agent is facing
    """

    holding: chex.Array  # ()
    job: chex.Array  # () - int: 0 if the agent is an apple collector, 1 if the agent is a basket transporter
    orientation: chex.Array  # () - float: the angle in radians that the agent is facing

# This is replacing Food class.
@dataclass
class ComplexOrchardApple(ComplexOrchardEntity):
    """
    The desired collectable.
    """
    held: chex.Array # () - bool: if the apple has been picked up by a bot
    collected: chex.Array # () - bool: if the apple has been dropped in a basket

@dataclass
class ComplexOrchardTree(ComplexOrchardEntity):
    """
    Obstacle in the environment.
    """
    fertility: chex.Array # () - float: the fertility of the tree

@dataclass
class ComplexOrchardBasket(ComplexOrchardEntity):
    """
    Where to deposit the apples.
    """
    held: chex.Array # () - bool: if the basket has been picked up by a bot


# This is replacing State class
# agents were changed to bots and now directly calling the Entity class as opposed to an Agent subclass
# trees also added to the class with same logic
# apples replaces 'food_item'
# time replaces 'step_count'
@dataclass
class ComplexOrchardState:
    """
    Holds the state of the simple orchard using JAX fundementals
    """

    bots: ComplexOrchardBot # List of bots (pytree structure)
    trees: ComplexOrchardTree # List of trees (pytree structure)
    apples: ComplexOrchardApple # List of apples (pytree structure)
    baskets: ComplexOrchardBasket # List of baskets (pytree structure)
    time: chex.Array # ()
    key: chex.PRNGKey # (2,)

# this replaces the Observation class.
# agents_view now only holds two channels (x,y) but now applied to trees as well.
class SimpleOrchardObservation(NamedTuple):
    """
    The observation "seen" by the bots given to the neural network as the input layer.
    """
    # in the format (num_agents, [apple[i].position[0], apple[i].position[1], ..., bot[i].position[0], bot[i].position[1], ...])
    agents_view: chex.Array # (num_agents, 2 * (num_apples + num_trees + num_bots))
    action_mask: chex.Array # (num_agents, 7) since there are 7 actions in the simple env [IDLE, UP, DOWN, LEFT, RIGHT, PICK, DROP]
    time: chex.Array # ()
