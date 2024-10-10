import gymnasium
import gymnasium.wrappers.record_video
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
    num_envs=16,
    # render_mode="human",
)  # or any other environment id

val_env = gymnasium.wrappers.record_video.RecordVideo(
    gymnasium.make("VizdoomCorridor-v0", render_mode="rgb_array"),
    video_folder="videos/",
    disable_logger=True,
)

screen_shape = envs.single_observation_space["screen"].shape
screen_shape = (screen_shape[2], screen_shape[0], screen_shape[1])
action_dim = envs.single_action_space.n

agent = Agent(screen_shape, action_dim)

try:
    agent.load_state_dict(torch.load("agent.pth"))
    print("Loaded agent from file")
except FileNotFoundError:
    print("No agent file found, starting from scratch")

optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Define the loss criterion


target_agent = Agent(screen_shape, action_dim).to(device)
target_agent.load_state_dict(agent.state_dict())


# ε-greedy
epsilon_start = 1.00
epsilon_final = 0.01
epsilon_decay = 5000
epsilon = epsilon_start
gamma = 0.99

batch_size = 256
start_training_after = batch_size * 2
target_update_frequency = 200

validation_frequency = 1000

print_frequency = 10
save_frequency = 100
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


def continuous_training(max_steps=100_000_000):
    observation, _ = envs.reset()
    step = 0
    loss = 0

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
        update_epsilon(step)

        if step % print_frequency == 0:
            print(f"Step: {step}, Loss: {loss:.8f}, Epsilon: {epsilon:.3f}")

        if step % target_update_frequency == 0:
            target_agent.load_state_dict(agent.state_dict())

        if step % save_frequency == 0 and step > 0:
            torch.save(agent.state_dict(), f"agent.pth")

        # Reset environment if all terminated or truncated
        if np.all(terminateds) or np.all(truncateds):
            observation, _ = envs.reset()

        if step % validation_frequency == 0:
            val_observation, _ = val_env.reset()

            while True:
                val_observation["screen"] = [val_observation["screen"]]
                val_state = augment_observation(val_observation)
                val_policy = agent(val_state)
                val_actions = val_policy.argmax().item()
                val_observation, _, val_terminated, val_truncated, _ = val_env.step(
                    val_actions
                )
                if val_terminated or val_truncated:
                    break


if __name__ == "__main__":

    print(f"Using device: {torch.get_default_device()}")

    continuous_training()
