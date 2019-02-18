import unittest

import torch

from machine.metrics import FrobeniusNorm


class TestDisruptiveness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.A = torch.tensor([[1,1],[1,1]])
        self.B = torch.tensor([[0,1],[1,0]])
        # A - B = eye(2)

        self.ans = torch.sqrt(torch.tensor(2, dtype=torch.float))

    def test_fn(self):
        fn = FrobeniusNorm()
        fn.eval_batch(self.A, self.B)
        val = fn.get_val()
        self.assertEqual(val, self.ans)

