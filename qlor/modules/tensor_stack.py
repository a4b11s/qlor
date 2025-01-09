import torch


class TensorStack:
    def __init__(self, stack_depth, batch_size, screen_shape):
        """
        TensorStack for stacking screens along the channel dimension for batched inputs.

        Args:
            stack_depth (int): Number of frames to stack.
            batch_size (int): Number of environments or batch size.
            screen_shape (tuple): Shape of a single screen (e.g., (C, H, W)).
        """
        self.stack_depth = stack_depth
        self.batch_size = batch_size
        self.screen_shape = screen_shape
        self.current_channels = screen_shape[0]
        self.total_channels = stack_depth * self.current_channels
        self.stack = torch.zeros(
            (batch_size, self.total_channels, *screen_shape[1:]), dtype=torch.float32
        )

    def add_batch(self, screens, dones):
        """
        Add a batch of screens and stack along the channel dimension. Reset the stack for environments where done is True.

        Args:
            screens (torch.Tensor): Batch of screens, shape (batch_size, C, H, W).
            dones (torch.Tensor): Batch of done flags, shape (batch_size,).
        """
        self._validate_screens(screens, dones)

        # Shift stack along the channel dimension
        self.stack = torch.roll(self.stack, shifts=-self.current_channels, dims=1)

        # Update the last channels in the stack with new screens
        self.stack[:, -self.current_channels :] = screens

        # Reset the stack for environments where done is True
        reset_mask = dones.view(-1, 1, 1, 1).expand(
            -1, self.total_channels, *self.screen_shape[1:]
        )
        self.stack = torch.where(reset_mask, torch.zeros_like(self.stack), self.stack)

    def get(self):
        """
        Retrieve the current stack of screens.

        Returns:
            torch.Tensor: The stacked screens, shape (batch_size, total_channels, H, W).
        """
        return self.stack

    def clear(self):
        """
        Clear the stack, resetting all frames to zeros.
        """
        self.stack = torch.zeros(
            (self.batch_size, self.total_channels, *self.screen_shape[1:]),
            dtype=torch.float32,
        )

    def __len__(self):
        return self.stack_depth

    def _validate_screens(self, screens, dones):
        """
        Validate input screens.

        Args:
            screens (torch.Tensor): Batch of screens, shape (batch_size, C, H, W).
            dones (torch.Tensor): Batch of done flags, shape (batch_size,).

        Raises:
            ValueError: If the input screens have an invalid shape.
        """

        if dones.shape != (self.batch_size,):
            raise ValueError(
                f"Expected dones of shape {(self.batch_size,)}, got {dones.shape}"
            )
        if screens.shape != (self.batch_size, *self.screen_shape):
            raise ValueError(
                f"Expected screens of shape {(self.batch_size, *self.screen_shape)}, got {screens.shape}"
            )


if __name__ == "__main__":
    stack_depth = 4
    batch_size = 4
    screen_shape = (1, 64, 64)

    # Initialize TensorStack
    tensor_stack = TensorStack(stack_depth, batch_size, screen_shape)

    # Simulate adding screens
    for step in range(3):
        screens = torch.randn((batch_size, *screen_shape))  # Random screens for testing
        tensor_stack.add_batch(screens)
        print(f"Stack after step {step + 1}:", tensor_stack.get().shape)
