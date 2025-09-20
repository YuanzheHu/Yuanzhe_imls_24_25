import gymnasium as gym
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def evaluate_blackjack_agent(agent, env, n_episodes:int)->None:
    """
    Evaluate the agent by counting the wins over n_episodes.

    Args:
        agent: The Q-learning agent
        env: The training environment
        n_episodes(int): The number of episodes on which to evaluate the agent

    Returns:
        None

    """
    wins = 0
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset() # reset the environment and get the initial observation
        done = False

        # play one episode
        while not done:
            # get the action from the agent
            action = agent.get_action(obs)

            # take a step in the environment using the action with BlackjackEnv.step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update if the environment is done
            done = terminated or truncated

        # evaluate if the agent won the game
        if reward >= 1:
            wins += 1
    return wins/n_episodes


def unpack_qvalues_dict(q_values: dict) -> np.ndarray:
    """
    Unpacks the Q-values dictionary into a numpy array.

    Args:
        q_values(defaultdict): The Q-values dictionary

    Returns:
        The Q-values in a numpy array
    """
    q_values_array = np.zeros((32, 11, 2))
    for player_sum in range(4, 32):
        for dealer_card in range(1, 11):
            for usable_ace in range(2):
                q_values_array[player_sum, dealer_card, usable_ace] = np.argmax(
                    q_values[(player_sum, dealer_card, usable_ace)]
                )
    return q_values_array


def get_best_actions_from_DQN(agent: tf.keras.Model, double_DQN=False) -> np.ndarray:
    """
    Get the action associated with the highest Q-value for each possible observation
    from the DQN agent.

    Args:
        agent (tf.keras.Model): The DQN agent.
        double_DQN (bool): If True, retrieves actions from the main network in Double DQN.

    Returns:
        np.ndarray: A (28, 10, 2) array where each entry contains the action (0 or 1)
                    with the highest Q-value for the corresponding state.
    """
    best_actions = np.zeros((28, 10, 2))  # Correct shape: (player_sum: 4-31, dealer: 1-10, ace: 0/1)
    
    # Generate all possible (player_sum, dealer_card, usable_ace) states
    states = np.array([
        [player_sum, dealer_card, usable_ace] 
        for player_sum in range(4, 32) 
        for dealer_card in range(1, 11) 
        for usable_ace in range(2)
    ], dtype=np.float32)  # Ensure float32 for TensorFlow

    # Get Q-value predictions from the correct model
    if double_DQN:
        predictions = agent.mainDQN.predict(states, verbose=0)
    else:
        predictions = agent.DQN.predict(states, verbose=0)

    # Store best action for each state
    for player_sum in range(4, 32):
        for dealer_card in range(1, 11):
            for usable_ace in range(2):
                index = (player_sum - 4) * 10 * 2 + (dealer_card - 1) * 2 + usable_ace
                best_actions[player_sum - 4, dealer_card - 1, usable_ace] = np.argmax(predictions[index])

    return best_actions


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int):
        """Initialize the experience replay buffer.

        Args:
            buffer_size(int): The maximum size of the buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.full = False

    def add(self, experience: tuple):
        """
        Add an experience to the buffer.

        Args:
            experience(tuple): The experience tuple to add
                experience[0](tuple): The observation
                experience[1](int): The action
                experience[2](float): The reward
                experience[3](bool): Whether the episode has terminated
                experience[4](tuple): The next observation
        """
        if len(self.buffer) >= self.buffer_size:
            self.full = True
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size(int): The size of the batch to sample

        Returns:
            A list of experiences
        """
        samples = np.random.choice(len(self.buffer), batch_size)
        obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch = zip(*[self.buffer[i] for i in samples])
        return np.array(obs_batch).reshape(batch_size, 3).astype(np.float32), np.array(action_batch).reshape(batch_size, -1), np.array(reward_batch).reshape(batch_size, -1), np.array(terminated_batch).reshape(batch_size, -1), np.array(next_obs_batch).reshape(batch_size, 3).astype(np.float32)
