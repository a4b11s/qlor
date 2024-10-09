import gymnasium
from vizdoom import gymnasium_wrapper  # This import will register all the environments
from qlor.agent import Agent
from torch import nn, optim, tensor
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

env = gymnasium.make("VizdoomBasic-v0", render_mode="human")  # or any other environment id

screen_shape = env.observation_space["screen"].shape
action_dim = env.action_space.n

agent = Agent(screen_shape, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Define the loss criterion


def augment_observation(observation):
    screen = observation["screen"]
    screen = tensor(screen).unsqueeze(0)
    return  screen / 255.0


def train(max_episodes=1000):
    for episode in range(max_episodes):
        agent.train()
        observation, _ = env.reset()
        print('-'*50)
        step = 0
        while True:
            if episode % 10 == 0:
                env.render()
            policy = agent(augment_observation(observation))
            action = policy.argmax().item()
            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            optimizer.zero_grad()
            reward_tensor = tensor([reward], dtype=torch.float32)
            loss = criterion(policy[0][action], reward_tensor)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters

            step += 1
            print(
                f"E: {episode}, S: {step}, A: {action}, R: {reward}, L: {loss.item()}\r", end=""
            )
            # your training code here


if __name__ == "__main__":
    
    print(f"Using device: {torch.get_default_device()}")
    
    train()
