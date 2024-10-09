import gymnasium
from vizdoom import gymnasium_wrapper  # This import will register all the environments
from qlor.agent import Agent
from torch import nn, optim, tensor
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

env = gymnasium.make("VizdoomBasic-v0")  # or any other environment id

screen_shape = env.observation_space["screen"].shape
action_dim = env.action_space.n

agent = Agent(screen_shape, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Define the loss criterion
gamma = 0.99


def augment_observation(observation):
    screen = observation["screen"]
    screen = tensor(screen).unsqueeze(0)
    return screen / 255.0


def train(max_episodes=1000):
    for episode in range(max_episodes):
        agent.train()
        observation, _ = env.reset()
        print("-" * 50)

        step = 0
        while True:
            # Отримання поточного стану
            current_state = augment_observation(observation)

            # Обчислення політики (Q-значення для всіх дій)
            policy = agent(current_state)

            # Вибір дії
            action = policy.argmax().item()

            # Виконання дії
            next_observation, reward, terminated, truncated, _ = env.step(action)

            # Перевірка на завершення епізоду
            if terminated or truncated:
                break

            # Обробка наступного стану
            next_state = augment_observation(next_observation)

            # Обчислення Q-значень для наступного стану
            with torch.no_grad():
                next_policy = agent(next_state)
                max_next_q_value = next_policy.max().item()  # max Q(s_{t+1}, a)

            # Цільове значення для поточного стану
            target_q_value = (
                reward + gamma * max_next_q_value
            )  # r + γ * max(Q(s_{t+1}, a))

            # Обчислення поточного Q-значення для обраної дії
            current_q_value = policy[0][action]

            # Обчислення втрати (loss)
            loss = criterion(current_q_value, tensor([target_q_value], device=device))

            # Оновлення моделі
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0:
                print(
                    f"E: {episode}, S: {step}, A: {action}, R: {reward}, L: {loss.item()}\r", end=""
                )


if __name__ == "__main__":

    print(f"Using device: {torch.get_default_device()}")

    train()
