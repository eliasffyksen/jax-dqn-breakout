from dataclasses import dataclass
from functools import partial
import random
from pprint import pprint

import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from wdb import get_wdb

from agent import Agent
from replay_buffer import ReplayBuffer
from policy import NatureCNN
from wandb_wrapper import WandbWrapper
from profiler import Profiler
from env.atari_wrappers import AtariWrapper
from env.vec_env.dummy_vec_env import DummyVecEnv
from env.vec_env.vec_frame_stack import VecFrameStack

@dataclass
class Hyperparameters:
    name: str = 'jax-dqn-breakout'
    gamma: float = 0.99
    grad_clip: float = 10.0
    buffer_size: int = 100_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    update_target_every: int = 1_000
    rollout_freq: int = 1
    train_freq: int = 4
    seed: int = 42
    train_length: int = 5_000_000
    log_interval: int = 1000
    eps_min: float = 0.01
    eps_max: float = 1.0
    eps_decay_len: float = 1_000_000
    start_learning: int = 100_000
    frame_stack: int = 4

def convert_state(state: jax.Array) -> jax.Array:
    return state.astype(jnp.float32) / 255.0

def calculate_epsilon(step: int, hyperparameters: Hyperparameters) -> float:
    eps_diff = hyperparameters.eps_max - hyperparameters.eps_min
    eps_decay_steps = hyperparameters.eps_decay_len
    eps_slope = eps_diff / eps_decay_steps
    eps = max(hyperparameters.eps_min, hyperparameters.eps_max - eps_slope * step)
    return eps

hyperparameters = Hyperparameters()
wdb = get_wdb(hyperparameters)

rng = jax.random.PRNGKey(hyperparameters.seed)
rng, agent_rng = jax.random.split(rng)
policy = NatureCNN()
optimiser = optax.chain(
    optax.clip_by_global_norm(hyperparameters.grad_clip),
    optax.adam(hyperparameters.learning_rate),
)

env = gym.make(
    "ALE/Breakout-v5",
    frameskip=1,
    repeat_action_probability=0.0,
)
if wdb is not None:
    env = WandbWrapper(env, wdb, 4000)
env = AtariWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, hyperparameters.frame_stack)
state = env.reset()
state = jnp.array(state[0])

agent = Agent(
    policy=policy,
    optimiser=optimiser,
    rng=agent_rng,
    state=state,
    gamma=hyperparameters.gamma,
)

replay_buffer = ReplayBuffer(
    capacity=hyperparameters.buffer_size,
    states=state,
    actions=jax.ShapeDtypeStruct((1,), jnp.int32),
    rewards=jax.ShapeDtypeStruct((1,), jnp.float32),
    next_states=state,
    terminals=jax.ShapeDtypeStruct((1,), jnp.bool_),
)

profiler = Profiler()

for step, rng in enumerate(jax.random.split(rng, hyperparameters.train_length)):

    if step % hyperparameters.rollout_freq == 0:
        profiler.enter('rollout.action_selection')

        eps = calculate_epsilon(step, hyperparameters)
        if random.uniform(0, 1) < eps or step < hyperparameters.start_learning:
            action = random.randint(0, 3)
        else:
            action = agent.predict(convert_state(state)[None])[0]
            action = action.argmax()

        profiler.enter('rollout.env_step')
        next_state, reward, terminal, _ = env.step([action])
        next_state = jnp.array(next_state[0])
        profiler.enter('rollout.save_to_buffer')
        replay_buffer.push(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            terminals=terminal,
        )
        state = next_state

        profiler.enter('rollout.reset')
        if terminal:
            state = env.reset()
            state = jnp.array(state[0])

    if step % hyperparameters.train_freq == 0:
        profiler.enter('train.sample')
        batch = replay_buffer.batch(hyperparameters.batch_size, rng)

        if batch is not None:
            profiler.enter('train.do')
            batch['states'] = convert_state(batch['states'])
            batch['next_states'] = convert_state(batch['next_states'])
            agent.train(batch)

    if step % hyperparameters.update_target_every == 0:
        profiler.enter('update_target')

        agent.update_target()

    if step % hyperparameters.log_interval == 0:
        profiler.enter('log')

        profiling_times = profiler.get()
        time_sum = sum(profiling_times.values())

        data = {
            'profiling': profiling_times,
            'profile_time_sum': time_sum,
            'time_per_frame': time_sum / max(step, 1),
        }

        if wdb is not None:
            wdb.log(data, step=step)
        else:
            pprint({
                'step': step,
                'data': data,
            })

    profiler.enter('misc')