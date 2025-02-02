import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNetwork(nn.Module):
    """
    A simple feed-forward neural network that estimates Q-values for each possible gadget recommendation.
    """

    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # first hidden layer
        self.fc2 = nn.Linear(64, 64)  # second hidden layer
        self.out = nn.Linear(64, action_size)  # output layer: one Q-value per gadget category

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """
    The DQN Agent that selects actions using an epsilon-greedy strategy,
    stores experiences in a replay buffer, and optimizes the policy network.
    """

    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=2000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor for future rewards
        self.epsilon = epsilon  # exploration rate (epsilon-greedy)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # Policy network and target network
        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        """
        Copies the weights from the policy network to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores a single transition in the replay buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """
        Returns an action using epsilon-greedy exploration.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def optimize_model(self):
        """
        Samples a batch from the replay buffer and performs a gradient descent step.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples yet

        # Sample a random batch from the replay buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Compute Q-values for current states using the policy network
        current_q = self.policy_net(states).gather(1, actions)
        # Compute the maximum Q-value for next states from the target network
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute the expected Q-values (target for training)
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def baseline_recommendation(personality):
    """
    Provides a baseline gadget recommendation (an action index) based on personality type.
    This is used for initial recommendations.
    """
    # Mapping of personality types (for young users) to gadget categories
    mapping = {
        "creative": 0,  # e.g., high-quality wireless headphones
        "gamer": 1,  # e.g., gaming consoles
        "tech-savvy": 2,  # e.g., smart home devices
        "minimalist": 3,  # e.g., streamlined minimalist gadgets
        "trendy": 4  # e.g., fashionable, trendy accessories
    }
    return mapping.get(personality, 0)  # default to 0 if personality is unknown


class GadgetEnv:
    """
    A simulated environment for gadget recommendations. The state vector encodes the personality (as a one-hot vector)
    and additional random features representing behavioral context.
    """

    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 5  # Five gadget categories
        self.personalities = ["creative", "gamer", "tech-savvy", "minimalist", "trendy"]

    def reset(self, personality=None):
        """
        Resets the environment with a given personality or selects one randomly.
        The initial state concatenates a one-hot personality encoding with random features.
        """
        if personality is None:
            self.personality = random.choice(self.personalities)
        else:
            self.personality = personality

        # Create a one-hot encoding for the personality (first 5 dimensions)
        personality_index = self.personalities.index(self.personality)
        personality_vector = np.zeros(len(self.personalities))
        personality_vector[personality_index] = 1

        # The remaining dimensions are random to simulate other behavioral features.
        random_features = np.random.rand(self.state_size - len(self.personalities))
        self.state = np.concatenate([personality_vector, random_features])
        return self.state

    def step(self, action):
        """
        Simulates taking an action in the environment.
        The reward is higher if the recommended gadget (action) matches the baseline recommendation for the personality.
        """
        baseline_action = baseline_recommendation(self.personality)
        if action == baseline_action:
            reward = np.random.uniform(0.8, 1.0)  # High reward if recommendation aligns
        else:
            reward = np.random.uniform(0.0, 0.5)  # Lower reward otherwise

        # Simulate the next state: keep the same personality (one-hot vector) and update the random features.
        personality_index = self.personalities.index(self.personality)
        personality_vector = np.zeros(len(self.personalities))
        personality_vector[personality_index] = 1
        random_features = np.random.rand(self.state_size - len(self.personalities))
        next_state = np.concatenate([personality_vector, random_features])

        # Simulate session termination: small chance to end the session after each step.
        done = np.random.rand() < 0.1  # 10% chance to end session
        return next_state, reward, done


def main():
    # Define dimensions:
    # - Use 5 dimensions to one-hot encode personality (creative, gamer, etc.)
    # - Add 5 additional random features.
    state_size = 10  # 5 (personality) + 5 (behavioral context)
    action_size = 5  # 5 gadget categories

    num_episodes = 500  # Number of training episodes (sessions)
    max_steps_per_episode = 50  # Maximum steps per session

    # Initialize the DQN Agent and the simulation environment.
    agent = DQNAgent(state_size, action_size)
    env = GadgetEnv(state_size)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()  # Reset environment: personality is randomly selected here.
        total_reward = 0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store the experience and optimize the agent’s policy.
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            agent.optimize_model()

            if done:
                break

        # Periodically update the target network for improved stability.
        if episode % 10 == 0:
            agent.update_target_network()

        if (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # --- Testing the trained agent ---
    # For demonstration, test the recommendation system for a fixed personality (e.g., "gamer").
    test_personality = "gamer"
    print(f"\nTesting recommendations for personality: {test_personality}")
    state = env.reset(personality=test_personality)

    # A list of gadget names corresponding to our 5 categories.
    recommended_gadgets = [
        "Wireless Headphones",  # creative
        "Gaming Console",  # gamer
        "Smart Home Device",  # tech-savvy
        "Minimalist Gadget",  # minimalist
        "Trendy Accessory"  # trendy
    ]

    for i in range(10):
        action = agent.select_action(state)
        print(f"Step {i + 1}: Recommended Gadget: {recommended_gadgets[action]}")
        state, reward, done = env.step(action)
        if done:
            break


if __name__ == "__main__":
    main()
