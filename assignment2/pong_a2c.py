import argparse
import datetime
import os
import random
import time

import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


def preprocess(image):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """

    image = torch.Tensor(image)

    # Crop, downsample by factor of 2, and turn to grayscale by keeping only red channel
    image = image[35:195]
    image = image[::2,::2, 0]

    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1

    return image.flatten().float()


def calc_discounted_future_rewards(rewards, discount_factor):
    r"""
    Calculate the discounted future reward at each timestep.

    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]

    """

    discounted_future_rewards = torch.empty(len(rewards))

    # Compute discounted_future_reward for each timestep by iterating backwards
    # from end of episode to beginning
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        if rewards[t] != 0:
            discounted_future_reward = 0

        # calculated discounted_future_reward at each timestep
        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards


class Actor(nn.Module):
    """ Simple two-layer MLP for policy network. """
    def __init__(self, input_size: int, hidden_size: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Critic(nn.Module):
    """ Simple two-layer MLP for value network. """
    def __init__(self, input_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def run_episode(actor, critic, env, discount_factor, render=False):
    """
        Plays an episode of the environment.
        Args:
            actor: the policy network
            critic: the state value function
            # model:
            env: the OpenAI environment
            discount_factor: discount_factor
            render: Render game window at 30fps
        Returns:
            loss:
            rewards: sum of the rewards for the episode
            # action_log_probs: log-probabilities of the takes actions in the trajectory
            state_values: the values of the states as calculated by the critic network
    """
    UP = 2
    DOWN = 3

    observation = env.reset()
    prev_x = preprocess(observation)

    action_chosen_log_probs = []
    values = []
    rewards = []

    done = False
    timestep = 0

    while not done:
        if render:
            # Render game window at 30fps
            time.sleep(1 / 30)
            env.render()

        # Preprocess the observation, set input to network to be difference
        # image between frames
        cur_x = preprocess(observation)
        x = cur_x - prev_x
        prev_x = cur_x

        # Run the actor network and sample action from the returned probability
        prob_up = actor(x)
        value = critic(x)
        action = UP if random.random() < prob_up else DOWN  # roll the dice!

        # Calculate the probability of sampling the action that was chosen
        action_chosen_prob = prob_up if action == UP else (1 - prob_up)
        action_chosen_log_probs.append(torch.log(action_chosen_prob))

        # Run the critic network to get the current state value
        # value = critic(x)
        values.append(value)

        # Step the environment, get new measurements, and updated discounted_reward
        observation, reward, done, info = env.step(action)
        rewards.append(torch.Tensor([reward]))
        timestep += 1

    # Concat lists of log probs and rewards into 1-D tensors
    action_chosen_log_probs = torch.cat(action_chosen_log_probs)
    rewards = torch.cat(rewards)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It causes roughly half of the actions to be encouraged and half to be discouraged, which
    # is helpful especially in beginning when +1 reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) \
                                     / discounted_future_rewards.std()

    # PG magic happens right here, multiplying action_chosen_log_probs by future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)

    # Calculate the loss that the optimizer will optimize
    actor_loss = -((discounted_future_rewards-value) * action_chosen_log_probs).sum()
    critic_loss = (discounted_future_rewards-value).pow(2).sum()

    return actor_loss, critic_loss, rewards.sum(), values


def train(render=False):
    # Hyperparameters
    input_size = 80 * 80 # input dimensionality: 80x80 grid
    hidden_size = 200 # number of hidden layer neurons
    learning_rate = 7e-4
    discount_factor = 0.99 # discount factor for reward

    batch_size = 4
    save_every_batches = 5

    # Create policy network
    # actor = MLP(input_size, hidden_size, 1)
    # critic = MLP(input_size, hidden_size, 1)
    actor = Actor(input_size, hidden_size)
    critic = Critic(input_size, hidden_size)

    # Load model weights and metadata from checkpoint if exists
    # if os.path.exists('checkpoint_a2c.pth'):
    #     print('Loading from checkpoint...')
    #     save_dict = torch.load('checkpoint_a2c.pth')
    #
    #     model.load_state_dict(save_dict['model_weights'])
    #     start_time = save_dict['start_time']
    #     last_batch = save_dict['last_batch']
    # else:
    start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
    last_batch = -1

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    # Set up tensorboard logging
    pt_writer = SummaryWriter(os.path.join('tensorboard_logs', start_time))

    # Create pong environment (PongDeterministic versions run faster)
    # Episodes consist of a series of games until one player has won 20 times.
    # A game ending in a win yields +1 reward and a game ending in a loss gives -1 reward.
    #
    # The RL agent (green paddle) plays against a simple AI (tan paddle) that
    # just tries to track the y-coordinate of the ball.
    env = gym.make("PongDeterministic-v4")

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    while True:

        mean_batch_loss = 0
        critic_batch_loss = 0
        mean_batch_reward = 0
        for batch_episode in range(batch_size):

            # Run one episode
            actor_loss, critic_loss, episode_reward, _ = run_episode(actor, critic, env, discount_factor, render)
            mean_batch_loss += actor_loss / batch_size
            critic_batch_loss += critic_loss / batch_size
            mean_batch_reward += episode_reward / batch_size

            # Boring book-keeping
            print(f'Episode reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        mean_batch_loss.backward(retain_graph=True)
        critic_batch_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()

        # Batch metrics and tensorboard logging
        print(f'Batch: {batch}, mean loss: {mean_batch_loss:.2f}, '
              f'mean reward: {mean_batch_reward:.2f}')
        pt_writer.add_scalar('mean loss', mean_batch_loss.item(), global_step=batch)
        pt_writer.add_scalar('mean reward', mean_batch_reward.item(), global_step=batch)
        # tensorboard --logdir=tensorboard_logs --port=6007

        # if batch % save_every_batches == 0:
        #     print('Saving checkpoint...')
        #     save_dict = {
        #         'model_weights': model.state_dict(),
        #         'start_time': start_time,
        #         'last_batch': batch
        #     }
        #     torch.save(save_dict, 'checkpoint_a2c.pth')

        batch += 1


def main():
    # By default, doesn't render game screen, but can invoke with `--render` flag on CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    train(render=args.render)


if __name__ == '__main__':
    main()
