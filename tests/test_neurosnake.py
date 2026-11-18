import tempfile
import unittest
from pathlib import Path
import numpy as np
from neurosnake.game import Snake
from neurosnake.network import NumpyNetwork


class GameTests(unittest.TestCase):
    def test_food_reward_and_growth(self):
        env = Snake()
        env.food = env.next_cell(0)
        state, reward, done = env.step(0)
        self.assertEqual((reward, done, env.score, len(env.body)), (10., False, 1, 4))
        self.assertEqual(state.shape, (11,))

    def test_wall_and_tail(self):
        env = Snake()
        env.body = [(11, 5), (10, 5), (9, 5)]
        self.assertEqual(env.step(0)[1:], (-10., True))
        env.reset()
        env.body = [(2, 2), (2, 3), (1, 3), (1, 2)]
        self.assertFalse(env.collision((1, 2)))
        self.assertTrue(env.collision((2, 3)))

    def test_full_board_and_timeout(self):
        env = Snake(6)
        env.body = [(x, y) for y in range(6) for x in range(6) if (x, y) != (1, 0)]
        env.food = (1, 0)
        self.assertTrue(env.step(0)[2])
        self.assertIsNone(env.food)
        env.reset()
        env.idle = 299
        env.food = (0, 0)
        self.assertEqual(env.step(0)[1:], (-10., True))


class NetworkTests(unittest.TestCase):
    def test_gradients_finite_difference(self):
        net = NumpyNetwork(hidden=5)
        x = np.random.default_rng(4).normal(size=(4, 11)).astype('f')
        a = np.array([0, 1, 2, 0]); y = np.array([.2, -.3, .4, .1])
        _, grads = net.gradients(x, a, y)
        for k, p in net.p.items():
            for index in np.ndindex(p.shape):
                original = p[index].copy(); eps = .002
                p[index] = original + eps
                plus = net.gradients(x, a, y)[0]
                p[index] = original - eps
                minus = net.gradients(x, a, y)[0]
                p[index] = original
                self.assertAlmostEqual(float(grads[k][index]), (plus-minus)/(2*eps), delta=.003)

    def test_learning_and_save(self):
        net = NumpyNetwork(hidden=8)
        x = np.eye(11, dtype='f')[:3]
        a = np.array([0, 1, 2]); targets = np.array([2., -1., 3.])
        before = net.gradients(x, a, targets)[0]
        for _ in range(1500):
            net.train(x, a, targets)
        self.assertLess(net.gradients(x, a, targets)[0], before*.1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.npz'
            net.save(path)
            restored = NumpyNetwork(hidden=8); restored.load(path)
            np.testing.assert_allclose(net.predict(x), restored.predict(x))


if __name__ == '__main__':
    unittest.main()
