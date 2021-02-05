import random
import time
import cv2
import gym
import numpy as np
import argparse
import pickle
import psutil
import sys
import tensorflow as tf
import traceback
import keras

from gym import spaces
from collections import deque
from math import isnan

# This part was based on
# https://github.com/openai/baselines/blob/edb52c22a5e14324304a491edc0f91b6cc07453b/baselines/common/atari_wrappers.py
# its license:
# The MIT License
# Copyright (c) 2017 OpenAI (http://openai.com)


cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_atari(env_id, max_episode_steps=400000):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


# This part was based on
# https://github.com/openai/baselines/blob/edb52c22a5e14324304a491edc0f91b6cc07453b/baselines/deepq/replay_buffer.py
# its license:
# The MIT License
# Copyright (c) 2017 OpenAI (http://openai.com)

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, goal, obs_t, action, reward, obs_tp1, done):
        data = (goal, obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        goals, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            goal, obs_t, action, reward, obs_tp1, done = data
            goals.append(np.array(goal, copy=False))
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(goals), np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(
            dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        goals: np.array
        obses_t: np.array
        actions: np.array
        rewards: np.array
        obses_tp1: np.array
        dones: np.array
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, name, value, step):
        # summary = tf.summary(value=[name, value])
        #
        # self.writer.add_summary(summary, step)

        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)


try:
    from gym.utils.play import play
except Exception as e:
    print("The following exception is typical for servers because they don't have display stuff installed. "
          "It only means that interactive --play won't work because `from gym.utils.play import play` failed with:")
    traceback.print_exc()
    print("You probably don't need --play on server, so let's continue.")

# Following main part was based on
# https://github.com/AdamStelmaszczyk/dqn
# Author: Adam Stelmaszczyk

DISCOUNT_FACTOR_GAMMA = 0.2
LEARNING_RATE = 0.0001
UPDATE_EVERY = 4
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 10000
TRAIN_START = 10000
REPLAY_BUFFER_SIZE = 100000
MAX_STEPS = 10000000
SNAPSHOT_EVERY = 500000
EVAL_EVERY = 100000
EVAL_STEPS = 20000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 100000
LOG_EVERY = 10000
VALIDATION_SIZE = 500
SIDE_BOXES = 4
BOX_PIXELS = 84 // SIDE_BOXES
STRATEGY = 'future'
K_EXTRA_GOALS = 4


def box_start(x):
    return (x // BOX_PIXELS) * BOX_PIXELS


def create_goal(position):
    goal = np.zeros(shape=(84, 84, 1))
    start_x, start_y = map(box_start, position)
    goal[start_x:start_x + BOX_PIXELS, start_y:start_y + BOX_PIXELS, 0] = 255
    return goal


def one_hot_encode(env, action):
    one_hot = np.zeros(env.action_space.n)
    one_hot[action] = 1
    return one_hot


def predict(env, model, goals, observations):
    frames_input = np.array(observations)
    actions_input = np.ones((len(observations), env.action_space.n))
    goals_input = np.array(goals)
    return model.predict([frames_input, actions_input, goals_input])


def save_for_debug(env, model, target_model, batch):
    model.save('model.h5')
    target_model.save('target_model.h5')
    pickle.dump((env, batch), open('debug.pkl', 'wb'))


def load_for_debug():
    model = keras.models.load_model('model.h5')
    target_model = keras.models.load_model('target_model.h5')
    env, batch = pickle.load(open('debug.pkl', 'rb'))
    return env, model, target_model, batch


def fit_batch(env, model, target_model, batch):
    goals, observations, actions, rewards, next_observations, dones = batch
    # Predict the Q values of the next states. Passing ones as the action mask.
    next_q_values = predict(env, target_model, goals, next_observations)
    # The Q values of terminal states is 0 by definition.
    next_q_values[dones] = 0.0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
    # Passing the actions as the mask and multiplying the targets by the actions masks.
    one_hot_actions = np.array([one_hot_encode(env, action) for action in actions])
    history = model.fit(
        x=[observations, one_hot_actions, goals],
        y=one_hot_actions * q_values[:, None],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    loss = history.history['loss'][0]
    if isnan(loss):
        save_for_debug(env, model, target_model, batch)
        print("loss is NaN, saved files for debug")
        sys.exit(1)
    return loss


def create_atari_model(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    print('n_actions {}'.format(n_actions))
    print(' '.join(env.unwrapped.get_action_meanings()))
    print('obs_shape {}'.format(obs_shape))
    frames_input = keras.layers.Input(obs_shape, name='frames_input')
    actions_input = keras.layers.Input((n_actions,), name='actions_input')
    goals_input = keras.layers.Input((84, 84, 1), name='goals_input')
    concatenated = keras.layers.concatenate([frames_input, goals_input])
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(concatenated)
    params = {
        'activation': 'relu',
    }
    conv_1 = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, **params)(normalized)
    conv_2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, **params)(conv_1)
    conv_3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, **params)(conv_2)
    conv_flattened = keras.layers.Flatten()(conv_3)
    hidden = keras.layers.Dense(512, **params)(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.multiply([output, actions_input])
    model = keras.models.Model([frames_input, actions_input, goals_input], filtered_output)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=0.1)
    model.compile(optimizer, loss='mae')
    return model


def epsilon_for_step(step):
    return max(EPSILON_FINAL, (EPSILON_FINAL - EPSILON_START) / EPSILON_STEPS * step + EPSILON_START)


def greedy_action(env, model, goal, observation):
    next_q_values = predict(env, model, goals=[goal], observations=[observation])
    return np.argmax(next_q_values)


def epsilon_greedy_action(env, model, goal, observation, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_action(env, model, goal, observation)
    return action


def save_model(model, step, logdir, name):
    filename = '{}/{}-{}.h5'.format(logdir, name, step)
    model.save(filename)
    print('Saved {}'.format(filename))
    return filename


def save_image(env, episode, step):
    frame = env.render(mode='rgb_array')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # following cv2.imwrite assumes BGR
    filename = "{}_{:06d}.png".format(episode, step)
    cv2.imwrite(filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])


def evaluate(env, model, view=False, images=False, eval_steps=EVAL_STEPS):
    done = True
    episode = 0
    episode_return_sum = 0.0
    episode_return_min = float('inf')
    episode_return_max = float('-inf')
    for step in range(1, eval_steps):
        if done:
            if episode > 0:
                print("eval episode {} steps {} return {}".format(
                    episode,
                    episode_steps,
                    episode_return,
                ))
                episode_return_sum += episode_return
                episode_return_min = min(episode_return_min, episode_return)
                episode_return_max = max(episode_return_max, episode_return)
            obs = env.reset()
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            goal = sample_goal()
            if view:
                env.render()
            if images:
                save_image(env, episode, step)
        else:
            obs = next_obs
        action = epsilon_greedy_action(env, model, goal, obs, EPSILON_FINAL)
        next_obs, _, done, _ = env.step(action)
        episode_return += goal_reward(next_obs, goal)
        episode_steps += 1
        if view:
            env.render()
        if images:
            save_image(env, episode, step)
    assert episode > 0
    episode_return_avg = episode_return_sum / episode
    return episode_return_avg, episode_return_min, episode_return_max


def find_agent(obs):
    image = obs[:, :, -1]
    indices = np.flatnonzero(image == 110)
    if len(indices) == 0:
        return None
    index = indices[0]
    x = index % 84
    y = index // 84
    return x, y


def goal_reward(obs, goal):
    agent_position = find_agent(obs)
    goal_reached = False
    if agent_position is not None:
        goal_reached = goal[agent_position] > 0
    return float(goal_reached)


def final_goal(trajectory):
    for experience in reversed(trajectory):
        _, _, _, _, next_obs, _ = experience
        agent = find_agent(next_obs)
        if agent:
            return create_goal(agent)
    return None


def future_goals(i, trajectory):
    goals = []
    if i + 1 >= len(trajectory):
        return None
    steps = np.random.randint(i + 1, len(trajectory), K_EXTRA_GOALS)
    for step in steps:
        _, _, _, _, next_obs, _ = trajectory[step]
        agent = find_agent(next_obs)
        if agent:
            goals.append(create_goal(agent))
    return goals


def sample_goal():
    position = np.random.randint(0, 84, 2)
    return create_goal(position)


def train(env, env_eval, model, max_steps, name):
    target_model = create_atari_model(env)
    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    done = True
    episode = 0
    logdir = '{}-log'.format(name)
    board = TensorBoardLogger(logdir)
    print('Created {}'.format(logdir))
    steps_after_logging = 0
    loss = 0.0
    for step in range(1, max_steps + 1):
        try:
            if step % SNAPSHOT_EVERY == 0:
                save_model(model, step, logdir, name)
            if done:
                if episode > 0:
                    if STRATEGY == 'final':
                        extra_goals = [final_goal(trajectory)]
                    for i, experience in enumerate(trajectory):
                        goal, obs, action, reward, next_obs, done = experience
                        replay.add(goal, obs, action, reward, next_obs, done)
                        # Hindsight Experience Replay - add experiences with extra goals that were reached
                        if STRATEGY == 'future':
                            extra_goals = future_goals(i, trajectory)
                        if extra_goals:
                            for extra_goal in extra_goals:
                                replay.add(extra_goal, obs, action, goal_reward(next_obs, extra_goal), next_obs, done)
                    if steps_after_logging >= LOG_EVERY:
                        steps_after_logging = 0
                        episode_end = time.time()
                        episode_seconds = episode_end - episode_start
                        episode_steps = step - episode_start_step
                        steps_per_second = episode_steps / episode_seconds
                        memory = psutil.virtual_memory()
                        to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
                        print(
                            "episode {} "
                            "steps {}/{} "
                            "loss {:.7f} "
                            "return {} "
                            "in {:.2f}s "
                            "{:.1f} steps/s "
                            "{:.1f}/{:.1f} GB RAM".format(
                                episode,
                                episode_steps,
                                step,
                                loss,
                                episode_return,
                                episode_seconds,
                                steps_per_second,
                                to_gb(memory.used),
                                to_gb(memory.total),
                            ))
                        board.log_scalar('episode_return', episode_return, step)
                        board.log_scalar('episode_steps', episode_steps, step)
                        board.log_scalar('episode_seconds', episode_seconds, step)
                        board.log_scalar('steps_per_second', steps_per_second, step)
                        board.log_scalar('epsilon', epsilon_for_step(step), step)
                        board.log_scalar('memory_used', to_gb(memory.used), step)
                        board.log_scalar('loss', loss, step)
                trajectory = []
                goal = sample_goal()
                episode_start = time.time()
                episode_start_step = step
                obs = env.reset()
                episode += 1
                episode_return = 0.0
                epsilon = epsilon_for_step(step)
            else:
                obs = next_obs
            action = epsilon_greedy_action(env, model, goal, obs, epsilon)
            next_obs, _, done, _ = env.step(action)
            reward = goal_reward(next_obs, goal)
            episode_return += reward
            trajectory.append((goal, obs, action, reward, next_obs, done))

            if step >= TRAIN_START and step % UPDATE_EVERY == 0:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.set_weights(model.get_weights())
                batch = replay.sample(BATCH_SIZE)
                loss = fit_batch(env, model, target_model, batch)
            if step == TRAIN_START:
                validation_goals, validation_observations, _, _, _, _ = replay.sample(VALIDATION_SIZE)
            if step >= TRAIN_START and step % EVAL_EVERY == 0:
                episode_return_avg, episode_return_min, episode_return_max = evaluate(env_eval, model)
                q_values = predict(env, model, validation_goals, validation_observations)
                max_q_values = np.max(q_values, axis=1)
                avg_max_q_value = np.mean(max_q_values)
                print(
                    "episode {} "
                    "step {} "
                    "episode_return_avg {:.1f} "
                    "episode_return_min {:.1f} "
                    "episode_return_max {:.1f} "
                    "avg_max_q_value {:.1f}".format(
                        episode,
                        step,
                        episode_return_avg,
                        episode_return_min,
                        episode_return_max,
                        avg_max_q_value,
                    ))
                board.log_scalar('episode_return_avg', episode_return_avg, step)
                board.log_scalar('episode_return_min', episode_return_min, step)
                board.log_scalar('episode_return_max', episode_return_max, step)
                board.log_scalar('avg_max_q_value', avg_max_q_value, step)
            steps_after_logging += 1
        except KeyboardInterrupt:
            save_model(model, step, logdir, name)
            break


def load_or_create_model(env, model_filename):
    if model_filename:
        model = keras.models.load_model(model_filename)
        print('Loaded {}'.format(model_filename))
    else:
        model = create_atari_model(env)
    model.summary()
    return model


def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed)


def print_weights(model):
    for layer in model.layers:
        weights_list = layer.get_weights()
        for weights in weights_list:
            print(np.array2string(weights, threshold=100000000))
        print()
        print('--------------------------------------------------------------------')
        print()


def main(args):
    assert BATCH_SIZE <= TRAIN_START <= REPLAY_BUFFER_SIZE
    assert TARGET_UPDATE_EVERY % UPDATE_EVERY == 0
    assert 84 % SIDE_BOXES == 0
    assert STRATEGY in ['final', 'future']
    print(args)
    env = make_atari('{}NoFrameskip-v4'.format(args.env))

    set_seed(env, args.seed)
    env_train = wrap_deepmind(env, frame_stack=True, episode_life=True, clip_rewards=True)
    if args.weights:
        model = load_or_create_model(env_train, args.model)
        print_weights(model)
    elif args.debug:
        env, model, target_model, batch = load_for_debug()
        fit_batch(env, model, target_model, batch)
    elif args.play:
        env = wrap_deepmind(env)
        play(env)
    else:
        env_eval = wrap_deepmind(env, frame_stack=True)
        model = load_or_create_model(env_train, args.model)
        if args.view or args.images or args.eval:
            evaluate(env_eval, model, args.view, args.images)
        else:
            max_steps = 100 if args.test else MAX_STEPS
            train(env_train, env_eval, model, max_steps, args.name)
            if args.test:
                filename = save_model(model, EVAL_STEPS, logdir='.', name='test')
                load_or_create_model(env_train, filename)


if __name__ == '__main__':
    MODEL_PATH = '/Users/liyang/Desktop/课件/I.A/Part_2/PART_2_PYTHON/hw/dqn_1/05-08-16-52-log/05-08-16-52-1000000.h5'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='load debug files and run fit_batch with them')
    parser.add_argument('--env', action='store', default='Freeway', help='Atari game name')
    parser.add_argument('--eval', action='store_true', default=False, help='run evaluation with log only')
    parser.add_argument('--images', action='store_true', default=True, help='save images during evaluation')
    parser.add_argument('--model', action='store', default=MODEL_PATH, help='model filename to load')
    parser.add_argument('--name', action='store', default=time.strftime("%m-%d-%H-%M"), help='name for saved files')
    parser.add_argument('--play', action='store_true', default=False, help='play with WSAD + Space')
    parser.add_argument('--seed', action='store', type=int, help='pseudo random number generator seed')
    parser.add_argument('--test', action='store_true', default=False, help='run tests')
    parser.add_argument('--view', action='store_true', default=False, help='view evaluation in a window')
    parser.add_argument('--weights', action='store_true', default=False, help='print model weights')
    main(parser.parse_args())
