import unittest

import numpy as np
import torch

from machine.metrics import FrobeniusNorm


class TestDisruptiveness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.A = torch.tensor([[1, 1], [1, 1]])
        self.B = torch.tensor([[0, 1], [1, 0]])
        # A - B = eye(2)

        self.state_1 = torch.tensor([[[0., 0., 0., 2., 2., 2., 2.],
                                      [0., 0., 0., 2., 1., 6., 1.],
                                      [0., 0., 0., 2., 1., 6., 1.],
                                      [0., 0., 0., 2., 1., 1., 1.],
                                      [0., 0., 0., 2., 1., 1., 1.],
                                      [0., 0., 0., 2., 1., 1., 1.],
                                      [0., 0., 0., 2., 7., 1., 7.]],

                                     [[0., 0., 0., 5., 5., 5., 5.],
                                      [0., 0., 0., 5., 0., 2., 0.],
                                      [0., 0., 0., 5., 0., 5., 0.],
                                      [0., 0., 0., 5., 0., 0., 0.],
                                      [0., 0., 0., 5., 0., 0., 0.],
                                      [0., 0., 0., 5., 0., 0., 0.],
                                      [0., 0., 0., 5., 3., 0., 0.]],

                                     [[0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.]]])

        self.state_2 = torch.tensor([[[0., 0., 0., 0., 2., 2., 2.],
                                      [0., 0., 0., 0., 2., 1., 6.],
                                      [0., 0., 0., 0., 2., 1., 6.],
                                      [0., 0., 0., 0., 2., 1., 1.],
                                      [0., 0., 0., 0., 2., 1., 1.],
                                      [0., 0., 0., 0., 2., 1., 1.],
                                      [0., 0., 0., 0., 2., 7., 1.]],

                                     [[0., 0., 0., 0., 5., 5., 5.],
                                      [0., 0., 0., 0., 5., 0., 2.],
                                      [0., 0., 0., 0., 5., 0., 5.],
                                      [0., 0., 0., 0., 5., 0., 0.],
                                      [0., 0., 0., 0., 5., 0., 0.],
                                      [0., 0., 0., 0., 5., 0., 0.],
                                      [0., 0., 0., 0., 5., 3., 0.]],

                                     [[0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.]]])

        self.ans = torch.sqrt(torch.tensor(2, dtype=torch.float))

        self.state_diff = torch.tensor([[[0., 0., 0., 2.,  0., 0.,  0.],
                                         [0., 0., 0., 2., -1., 5., -5.],
                                         [0., 0., 0., 2., -1., 5., -5.],
                                         [0., 0., 0., 2., -1., 0.,  0.],
                                         [0., 0., 0., 2., -1., 0.,  0.],
                                         [0., 0., 0., 2., -1., 0.,  0.],
                                         [0., 0., 0., 2.,  5., -6., 6.]],

                                        [[0., 0., 0., 5.,  0.,  0.,  0.],
                                         [0., 0., 0., 5., -5.,  2., -2.],
                                         [0., 0., 0., 5., -5.,  5., -5.],
                                         [0., 0., 0., 5., -5.,  0.,  0.],
                                         [0., 0., 0., 5., -5.,  0.,  0.],
                                         [0., 0., 0., 5., -5.,  0.,  0.],
                                         [0., 0., 0., 5., -2., -3.,  0.]],

                                        [[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]]])

        self.ans_2 = np.sqrt(230) + np.sqrt(371)

        # self.state_diff_binary = torch.nonzero(self.state_diff)
        self.ans_3 = 37

    def test_fn(self):
        fn = FrobeniusNorm()
        fn.eval_batch(self.A, self.B)
        val = fn.get_val()
        self.assertEqual(val, self.ans)

    def test_full_state(self):
        fn = FrobeniusNorm()
        fn.eval_batch(self.state_1, self.state_2)
        val = fn.get_val()
        self.assertAlmostEqual(val, self.ans_2, places=5)

    def test_full_binary(self):
        fn = FrobeniusNorm(binary=True)
        fn.eval_batch(self.state_1, self.state_2)
        val = fn.get_val()
        self.assertAlmostEqual(val, self.ans_3, places=5)
