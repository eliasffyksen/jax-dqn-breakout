from functools import partial
from typing import Callable

import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
import jaxopt


def make_train_fn(
    opt: optax.GradientTransformation,
    policy: nn.Module,
    gamma: float,
) -> Callable:
    def dqn_loss(
            policy_params: optax.Params,
            policy_target_params: optax.Params,
            states: jax.Array,
            actions: jax.Array,
            rewards: jax.Array,
            next_states: jax.Array,
            terminals: jax.Array,
    ) -> jax.Array:
        q_values = policy.apply(policy_params, states)
        q_values = jnp.take_along_axis(q_values, actions, axis=-1)[:, 0]
        q_target_values = policy.apply(policy_target_params, next_states)
        q_target_values = jnp.max(q_target_values, axis=-1, keepdims=True)
        q_target_values = rewards + (~terminals) * gamma * q_target_values
        loss = jaxopt.loss.huber_loss(q_values, q_target_values)

        return loss.mean()

    grad_fn = jax.grad(dqn_loss)

    def train_step(
            policy_params: optax.Params,
            opt_state: optax.OptState,
            policy_target_params: optax.Params,
            states: jax.Array,
            actions: jax.Array,
            rewards: jax.Array,
            next_states: jax.Array,
            terminals: jax.Array,
    ) -> tuple[optax.Params, optax.OptState]:
        print('tracing train step')
        grads = grad_fn(policy_params, policy_target_params,
                        states, actions, rewards, next_states, terminals)
        updates, opt_state = opt.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)

        return policy_params, opt_state

    return train_step

class Agent:
    policy_params: optax.Params
    policy_target_params: optax.Params
    opt: optax.GradientTransformation
    opt_state: optax.OptState

    def __init__(
        self,
        policy: nn.Module,
        optimiser: optax.GradientTransformation,
        state: jax.ShapeDtypeStruct,
        rng: jax.Array,
        gamma: float,
    ) -> None:
        self.policy = policy
        self.opt = optimiser

        self._setup_policy(state, rng)
        self._setup_optimiser()
        self._setup_train(gamma)
    
    def _setup_policy(self, state: jax.ShapeDtypeStruct, rng: jax.Array) -> None:
        dummy_data = jnp.zeros((1, *state.shape), dtype=state.dtype)
        self.policy_params = self.policy.init(rng, dummy_data)
        self._predict = jax.jit(self.policy.apply)

        self.update_target()

    def _setup_optimiser(self) -> None:
        self.opt_state = self.opt.init(self.policy_params)

    def _setup_train(self, gamma: float) -> None:
        self._train = jax.jit(
            make_train_fn(self.opt, self.policy, gamma),
            donate_argnames=("policy_params", "opt_state")
        )

    def update_target(self) -> None:
        self.policy_target_params = jax.tree_map(
            lambda x: x.copy(),
            self.policy_params,
        )

    def predict(self, state: jax.Array) -> jax.Array:
        return self._predict(self.policy_params, state)

    def train(self, batch: dict[str, jax.Array]) -> None:
        self.policy_params, self.opt_state = self._train(
            self.policy_params,
            self.opt_state,
            self.policy_target_params,
            **batch,
        )
