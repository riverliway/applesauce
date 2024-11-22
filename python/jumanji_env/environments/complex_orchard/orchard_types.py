### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/types.py ####

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from jumanji_env.environments.complex_orchard.constants import NUM_ACTIONS, JaxArray

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
    id: JaxArray # ()
    position: JaxArray[2]
    diameter: JaxArray # ()

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

    holding: JaxArray  # ()
    job: JaxArray  # () - int: 0 if the agent is an apple collector, 1 if the agent is a basket transporter
    orientation: JaxArray  # () - float: the angle in radians that the agent is facing

# This is replacing Food class.
@dataclass
class ComplexOrchardApple(ComplexOrchardEntity):
    """
    The desired collectable.
    """
    held: JaxArray # () - bool: if the apple has been picked up by a bot
    collected: JaxArray # () - bool: if the apple has been dropped in a basket

@dataclass
class ComplexOrchardTree(ComplexOrchardEntity):
    """
    Obstacle in the environment.
    """
    fertility: JaxArray # () - float: the fertility of the tree

@dataclass
class ComplexOrchardBasket(ComplexOrchardEntity):
    """
    Where to deposit the apples.
    """
    held: JaxArray # () - bool: if the basket has been picked up by a bot


# This is replacing State class
# agents were changed to bots and now directly calling the Entity class as opposed to an Agent subclass
# trees also added to the class with same logic
# apples replaces 'food_item'
@dataclass
class ComplexOrchardState:
    """
    Holds the state of the simple orchard using JAX fundementals
    """

    bots: ComplexOrchardBot # List of bots (pytree structure)
    trees: ComplexOrchardTree # List of trees (pytree structure)
    apples: ComplexOrchardApple # List of apples (pytree structure)
    baskets: ComplexOrchardBasket # List of baskets (pytree structure)
    step_count: JaxArray # ()
    width: JaxArray # ()
    height: JaxArray # ()
    key: chex.PRNGKey

class ComplexOrchardObservation(NamedTuple):
    """
    The observation "seen" by the bots given to the neural network as the input layer.
    """
    agents_view: JaxArray['num_bots', 'num_observations'] # the view of the agents, it is dependent on which observer is used
    action_mask: JaxArray['num_bots', 'NUM_ACTIONS'] # NUM_ACTIONS will be 7 since there are 7 actions [IDLE, UP, DOWN, LEFT, RIGHT, PICK, DROP]
    step_count: JaxArray # ()
