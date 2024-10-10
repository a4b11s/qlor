import gymnasium
import numpy as np
from vizdoom import gymnasium_wrapper  # This import will register all the environments
from qlor.agent import Agent
from torch import nn, optim, tensor
import torch
import random
import collections
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

envs = gymnasium.make_vec(
    "VizdoomCorridor-v0",  # or any other environment id
    num_envs=32,
    # render_mode="human",
)  # or any other environment id


screen_shape = envs.single_observation_space["screen"].shape
screen_shape = (screen_shape[2], screen_shape[0], screen_shape[1])
action_dim = envs.single_action_space.n

agent = Agent(screen_shape, action_dim)
agent.load_state_dict(torch.load("agent.pth"))

optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Define the loss criterion


target_agent = Agent(screen_shape, action_dim).to(device)
target_agent.load_state_dict(agent.state_dict())


# ε-greedy
epsilon_start = 0.01
epsilon_final = 0.01
epsilon_decay = 500
epsilon = epsilon_start
gamma = 0.99

batch_size = 256
start_training_after = batch_size * 2
target_update_frequency = 20

save_frequency = 100
clear_experience_replay_frequency = 10
experience_replay_maxlen = 2000

experience_replay = collections.deque(maxlen=experience_replay_maxlen)


def epsilon_greedy_action(policy, epsilon):
    actions = []
    for p in policy:
        actions.append(
            random.randrange(action_dim)
            if random.random() < epsilon
            else p.argmax().item()
        )
    return np.array(actions)


def update_epsilon(episode):
    global epsilon
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * episode / epsilon_decay
    )


def augment_observation(observation):
    screen = observation["screen"]
    screen = tensor(screen, device=device)  # bs w h c
    # From bshwc to bschw
    screen = screen.permute(0, 3, 1, 2)

    return screen / 255.0


def train_batch():
    batch_indices = np.random.choice(len(experience_replay), batch_size, replace=False)
    batch = [experience_replay[i] for i in batch_indices]

    state_batch = torch.cat([experience[0] for experience in batch]).to(device)
    action_batch = tensor(
        [experience[1] for experience in batch], device=device
    ).long()  # Ensure action is long type for indexing
    reward_batch = tensor(
        [experience[2] for experience in batch], device=device, dtype=torch.float32
    )  # Cast reward to float32
    next_state_batch = torch.cat([experience[3] for experience in batch]).to(device)
    done_batch = tensor(
        [experience[4] for experience in batch], device=device, dtype=torch.float32
    )  # Cast done flag to float32

    # Поточні Q-значення
    policy_batch = agent(state_batch)
    current_q_values = policy_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_policy_batch = agent(next_state_batch)
        next_actions = next_policy_batch.argmax(dim=1)
        target_policy_batch = target_agent(next_state_batch)
        next_q_values = target_policy_batch.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = criterion(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def episode_fn(episode, max_steps=1000):
    agent.train()
    observation, _ = envs.reset()
    update_epsilon(episode)
    step = 0
    while step < max_steps:
        torch.cuda.empty_cache()
        current_state = augment_observation(observation)
        policy = agent(current_state)
        actions = epsilon_greedy_action(policy, epsilon)
        next_observations, rewards, terminateds, truncateds, _ = envs.step(actions)

        for i in range(envs.num_envs):
            experience_replay.append(
                (
                    current_state[i].unsqueeze(0).cpu(),  # Single environment state
                    actions[i],
                    rewards[i],
                    augment_observation(next_observations)[i].unsqueeze(0).cpu(),
                    terminateds[i] or truncateds[i],
                )
            )

        if len(experience_replay) > start_training_after and step % 4 == 0:
            loss = train_batch()

        observation = next_observations
        step += 1

        if np.all(terminateds) or np.all(truncateds):
            break
    if episode % target_update_frequency == 0:
        target_agent.load_state_dict(agent.state_dict())

    if episode % save_frequency == 0 and episode > 0:
        torch.save(agent.state_dict(), f"agent.pth")


def train(max_episodes=1000):
    for episode in range(max_episodes):
        start_time = time.time()
        episode_fn(episode)
        print(
            f"\nEpisode {episode} took {time.time() - start_time:.2f} seconds. {epsilon:.3f} epsilon."
        )

    print("\nTraining completed.")


if __name__ == "__main__":

    print(f"Using device: {torch.get_default_device()}")

    train(1_000_000)
