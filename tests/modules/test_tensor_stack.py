import unittest
import torch
from qlor.modules.tensor_stack import TensorStack


class TestTensorStack(unittest.TestCase):
    def setUp(self):
        self.stack_depth = 3
        self.batch_size = 2
        self.screen_shape = (1, 4, 4)
        self.tensor_stack = TensorStack(
            self.stack_depth, self.batch_size, self.screen_shape
        )

    def test_initial_state(self):
        self.assertTrue(
            torch.all(self.tensor_stack.get() == 0), "Initial stack should be all zeros"
        )

    def test_add_single_batch(self):
        screens = torch.ones((self.batch_size, *self.screen_shape))
        dones = torch.zeros(self.batch_size, dtype=torch.bool)
        self.tensor_stack.add_batch(screens, dones)
        result = self.tensor_stack.get()
        self.assertTrue(
            torch.all(result[:, -1:] == 1), "Last added screens should be ones"
        )
        self.assertTrue(
            torch.all(result[:, :-1] == 0),
            "Other channels should remain zeros after one add_batch",
        )

    def test_stack_multiple_batches(self):
        screens1 = torch.ones((self.batch_size, *self.screen_shape))
        screens2 = torch.full((self.batch_size, *self.screen_shape), 2)
        dones = torch.zeros(self.batch_size, dtype=torch.bool)
        self.tensor_stack.add_batch(screens1, dones)
        self.tensor_stack.add_batch(screens2, dones)
        result = self.tensor_stack.get()
        self.assertTrue(
            torch.all(result[:, -2:-1] == 1), "Second to last channels should be ones"
        )
        self.assertTrue(torch.all(result[:, -1:] == 2), "Last channels should be twos")

    def test_rolling_behavior(self):
        screens1 = torch.ones((self.batch_size, *self.screen_shape))
        screens2 = torch.full((self.batch_size, *self.screen_shape), 2)
        screens3 = torch.full((self.batch_size, *self.screen_shape), 3)
        dones = torch.zeros(self.batch_size, dtype=torch.bool)
        self.tensor_stack.add_batch(screens1, dones)
        self.tensor_stack.add_batch(screens2, dones)
        self.tensor_stack.add_batch(screens3, dones)
        result = self.tensor_stack.get()
        self.assertTrue(
            torch.all(result[:, -3:-2] == 1), "Third to last channels should be ones"
        )
        self.assertTrue(
            torch.all(result[:, -2:-1] == 2), "Second to last channels should be twos"
        )
        self.assertTrue(
            torch.all(result[:, -1:] == 3), "Last channels should be threes"
        )

    def test_done_flags_behavior(self):
        screens1 = torch.ones((self.batch_size, *self.screen_shape))
        screens2 = torch.full((self.batch_size, *self.screen_shape), 2)
        done_flags = torch.tensor([False, True], dtype=torch.bool)
        self.tensor_stack.add_batch(screens1, done_flags)
        self.tensor_stack.add_batch(screens2, done_flags)
        result = self.tensor_stack.get()
        self.assertTrue(
            torch.all(result[0, -2:-1] == 1),
            "First batch second to last channels should be ones",
        )
        self.assertTrue(
            torch.all(result[0, -1:] == 2), "First batch last channels should be twos"
        )
        self.assertTrue(
            torch.all(result[1, -2:] == 0),
            "Second batch all channels should be zeros",
        )


if __name__ == "__main__":
    unittest.main()
