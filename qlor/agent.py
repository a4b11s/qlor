from torch import nn, Tensor
from torch.optim import Optimizer


class Agent(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super(Agent, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def train_on_batch(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        reward_batch: Tensor,
        next_state_batch: Tensor,
        done_batch: Tensor,
        optimizer: Optimizer,
        criterion: nn.Module,
        calculate_target_q_values: callable,
    ):
        self.train()

        # Current Q values
        policy_batch = self(state_batch)

        current_q_values = policy_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        target_q_values = calculate_target_q_values(
            next_state_batch, reward_batch, done_batch
        )

        loss = criterion(current_q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.eval()

        return loss.item()
