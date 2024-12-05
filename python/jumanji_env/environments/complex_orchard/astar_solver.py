from jumanji_env.environments.complex_orchard.env import ComplexOrchard
from jumanji_env.environments.complex_orchard.generator import ComplexOrchardGenerator
from apple_mava.log_dicts import create_complex_dict
from simulation.orchard import OrchardComplex2D
from simulation.complex_solver import ComplexSolver
import chex
import jax.numpy as jnp
import jax
import json
import os

from jumanji_env.environments.complex_orchard.constants import (
  JaxArray,
  LEFT,
  RIGHT,
  FORWARD,
  BACKWARD,
  NOOP,
  PICK,
  DROP,
)

class AStarSolver:
  def __init__(self, gen: ComplexOrchardGenerator, key: chex.PRNGKey):
    self.gen = gen
    self.env = ComplexOrchard(gen, time_limit=10_000)
    self.key = key
    self.id = jax.random.randint(key, (), 1, 10_000).item()
    self.state, _ = self.env.reset(self.key)

    self.write = True
    self.log_file = f'run-{self.id}.json'

    self._setup()

  def _setup(self):
    """
    Sets up the solver to solve the environment.
    """
    old_env = self._create_old_env()
    self.solver = ComplexSolver(old_env)

    if os.path.exists(self.log_file):
      os.remove(self.log_file)

  def _create_old_env(self) -> OrchardComplex2D:
    """
    Creates the old environment for the solver.
    """

    dict = create_complex_dict(self.state, 0)

    if self.write:
      with open(self.log_file, 'a') as f:
        f.write(json.dumps(dict))
        f.write(',\n')

    old_env = OrchardComplex2D(dict['width'], dict['height'], self.env.num_picker_bots, self.env.num_pusher_bots, self.env.num_baskets, 0)
    old_env.trees = dict['trees']
    old_env.baskets = dict['baskets']
    old_env.apples = dict['apples']
    old_env.bots = dict['bots']

    return old_env

  def simulate(self) -> int:
    """
    Runs the solver until it finishes.
    This cannot be VMAPed as the solver is stateful.

    :return: The number of steps taken to solve the environment.
    """

    did_finish = False

    while not did_finish:
      did_finish = self.step()

    return self.env.step_count

  def step(self) -> bool:
    """
    Perform a step of the solver.

    :return: True if the solver has finished, False otherwise.
    """

    state, timestep = self.env.step(self.state, self._solve())
    self.state = state

    # print(timestep.extras['percent_collected'])

    return timestep.extras['percent_collected'] >= 95

  def _solve(self) -> JaxArray['num_agents']:
    """
    Determines what the next action should be for all of the bots

    :return: The action to take for each bot
    """
  
    self.solver.environment = self._create_old_env()

    actions = self.solver.make_decisions()
    print(actions)

    return jnp.array(
      [self._convert_action(action) for action in actions],
      dtype=jnp.int32
    )
  
  def _convert_action(self, action: str) -> int:
    """
    Converts the string action to the integer action.

    :param action: The string action to convert.

    :return: The integer action.
    """

    if action == 'left':
      return LEFT
    elif action == 'right':
      return RIGHT
    elif action == 'forward':
      return FORWARD
    elif action == 'backward':
      return BACKWARD
    elif action == 'idle':
      return NOOP
    elif action == 'pick':
      return PICK
    elif action == 'drop':
      return DROP
    
    raise ValueError(f"Invalid action: {action}")