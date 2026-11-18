"""Two-layer Q network: forward, Huber loss and gradients by hand."""
import numpy as np


class NumpyNetwork:
    def __init__(self, seed=42, hidden=128):
        rng = np.random.default_rng(seed)
        self.p = dict(w1=rng.normal(0, np.sqrt(2/11), (11, hidden)).astype('f'),
                      b1=np.zeros(hidden, 'f'),
                      w2=rng.normal(0, np.sqrt(2/hidden), (hidden, 3)).astype('f'),
                      b2=np.zeros(3, 'f'))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def predict(self, x):
        return np.maximum(0, np.asarray(x) @ self.p['w1'] + self.p['b1']) @ self.p['w2'] + self.p['b2']

    def gradients(self, x, actions, targets):
        z = x @ self.p['w1'] + self.p['b1']
        h = np.maximum(0, z)
        q = h @ self.p['w2'] + self.p['b2']
        error = q[np.arange(len(x)), actions] - targets
        absolute = np.abs(error)
        loss = np.mean(np.where(absolute <= 1, .5 * error**2, absolute-.5))
        dq = np.zeros_like(q)
        dq[np.arange(len(x)), actions] = np.clip(error, -1, 1) / len(x)
        dh = (dq @ self.p['w2'].T) * (z > 0)
        return float(loss), dict(w1=x.T @ dh, b1=dh.sum(0), w2=h.T @ dq, b2=dq.sum(0))

    def train(self, x, actions, targets, lr=.001):
        loss, grads = self.gradients(x, actions, targets)
        self.t += 1
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        for k, g in grads.items():
            g = g * min(1., 10 / (norm + 1e-8))
            self.m[k] = .9*self.m[k] + .1*g
            self.v[k] = .999*self.v[k] + .001*g*g
            self.p[k] -= lr * (self.m[k]/(1-.9**self.t)) / (np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
        return loss

    def copy_from(self, other):
        self.p = {k: v.copy() for k, v in other.p.items()}

    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        with np.load(path) as data:
            self.p = {k: data[k].copy() for k in self.p}


class TorchNetwork:
    def __init__(self, seed=42):
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(11, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

    def predict(self, x):
        with self.torch.no_grad():
            return np.array(self.model(self.torch.tensor(np.asarray(x).tolist(), dtype=self.torch.float32)).tolist(), dtype='f')

    def train(self, x, actions, targets):
        t = self.torch
        q = self.model(t.tensor(x.tolist(), dtype=t.float32))
        selected = q[t.arange(len(x)), t.tensor(actions.tolist())]
        loss = t.nn.functional.smooth_l1_loss(selected, t.tensor(targets.tolist(), dtype=t.float32))
        self.optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        return loss.item()

    def copy_from(self, other):
        self.model.load_state_dict(other.model.state_dict())

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(self.torch.load(path, map_location='cpu', weights_only=True))
