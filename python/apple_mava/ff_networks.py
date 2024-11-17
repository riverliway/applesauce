from typing import Sequence  # For typing hints
import chex  # For type checking arrays
import jax.numpy as jnp  # For numerical operations
import numpy as np  # For numerical operations
from flax import linen as nn  # For defining neural networks
from jax.nn.initializers import orthogonal  # For orthogonal weight initialization
import tensorflow_probability.substrates.jax.distributions as tfd  # For probabilistic distributions
from mava.types import (
    Observation,
    ObservationGlobalState,
)
from mava.networks.distributions import IdentityTransformation

class Actor(nn.Module):
    """The `Actor()` network takes an observation as input and produces logits representing the probabilities of different actions. The shapes within the network are determined dynamically based on the number of agents, the observation, and the batch size."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, observation: Observation) -> tfd.TransformedDistribution:
        """Forward pass."""
        x = observation.agents_view

        actor_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(x)
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(actor_output)
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))


class Critic(nn.Module):
    """The `Critic()` network takes the global state as input and produces the estimated value of the state. Similar to the Actor network, the shapes within the network are handled implicitly by Flax."""

    @nn.compact
    def __call__(self, observation: ObservationGlobalState) -> chex.Array:
        """Forward pass."""
        critic_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(observation.agents_view)
        critic_output = nn.relu(critic_output)
        critic_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(critic_output)
        critic_output = nn.relu(critic_output)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return jnp.squeeze(critic_output, axis=-1)