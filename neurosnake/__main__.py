import argparse
import csv
import json
from collections import deque
from pathlib import Path
import random
import numpy as np
from .game import Snake
from .network import NumpyNetwork, TorchNetwork


def network(backend, seed):
    return (NumpyNetwork if backend == 'numpy' else TorchNetwork)(seed)


def train(args):
    rng = random.Random(args.seed)
    env = Snake(args.size, args.seed)
    online = network(args.backend, args.seed)
    target = network(args.backend, args.seed)
    target.copy_from(online)
    memory = deque(maxlen=50_000)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scores, rows, steps = [], [], 0
    suffix = '.npz' if args.backend == 'numpy' else '.pt'
    with (out / 'config.json').open('w') as f:
        json.dump(vars(args), f, indent=2)
    for episode in range(1, args.games+1):
        state = env.reset()
        epsilon = max(.02, 1 - (episode-1) / max(1, args.games*.7))
        losses = []
        while not env.done:
            action = rng.randrange(3) if rng.random() < epsilon else int(online.predict(state).argmax())
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1
            if len(memory) >= args.batch and steps % 4 == 0:
                batch = rng.sample(list(memory), args.batch)
                s, a, r, ns, d = map(np.array, zip(*batch))
                # Double DQN: choose with online, evaluate with frozen target.
                next_actions = online.predict(ns).argmax(axis=1)
                future = target.predict(ns)[np.arange(len(ns)), next_actions]
                y = r + .95 * (~d) * future
                losses.append(online.train(s, a, y))
            if steps % 500 == 0:
                target.copy_from(online)
        scores.append(env.score)
        avg = float(np.mean(scores[-100:]))
        rows.append((episode, env.score, avg, epsilon, float(np.mean(losses)) if losses else 0))
        if episode % 25 == 0 or episode == args.games:
            print(f'Game {episode:5d} | score {env.score:3d} | mean(100) {avg:.2f} | epsilon {epsilon:.3f}', flush=True)
            online.save(out / ('model'+suffix))
    with (out / 'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game', 'score', 'mean_100', 'epsilon', 'loss'])
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(scores, alpha=.35, label='Score')
    plt.plot([r[2] for r in rows], label='Rolling mean (100)')
    plt.xlabel('Game'); plt.ylabel('Food eaten'); plt.legend(); plt.tight_layout()
    plt.savefig(out / 'learning_curve.png'); plt.close()
    print(f'Saved model, scores and learning curve in {out}')


def evaluate(args):
    agent = network(args.backend, args.seed)
    agent.load(args.model)
    scores = []
    for i in range(args.games):
        env = Snake(args.size, args.seed+i)
        while not env.done:
            env.step(int(agent.predict(env.state()).argmax()))
        scores.append(env.score)
    result = dict(games=args.games, seed=args.seed, board_size=args.size,
                  mean=float(np.mean(scores)), std=float(np.std(scores)),
                  best=max(scores), scores=scores, exploration=False)
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))


def play(args):
    import pygame as pg
    pg.init()
    screen = pg.display.set_mode((600, 680))
    pg.display.set_caption('NeuroSnake | NumPy Deep Q-Learning')
    font = pg.font.SysFont('consolas', 20)
    clock = pg.time.Clock()
    env = Snake(args.size, args.seed)
    agent = network(args.backend, args.seed) if args.model else None
    if agent:
        agent.load(args.model)
    paused = False
    desired = env.direction
    running = True
    cell = 560 / args.size
    try:
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_SPACE:
                        paused = not paused
                    elif event.key == pg.K_r:
                        env.reset(); desired = env.direction
                    elif event.key in (pg.K_RIGHT, pg.K_DOWN, pg.K_LEFT, pg.K_UP):
                        new = (pg.K_RIGHT, pg.K_DOWN, pg.K_LEFT, pg.K_UP).index(event.key)
                        if (new-env.direction) % 4 != 2:
                            desired = new
            if running and not env.done and not paused:
                if agent:
                    action = int(agent.predict(env.state()).argmax())
                else:
                    delta = (desired-env.direction) % 4
                    action = {0: 0, 1: 1, 3: 2}.get(delta, 0)
                env.step(action)
            screen.fill('#0b1220')
            for x in range(args.size):
                for y in range(args.size):
                    pg.draw.rect(screen, '#162238', (20+x*cell, 80+y*cell, cell-1, cell-1), 1)
            for i, (x, y) in enumerate(env.body):
                pg.draw.rect(screen, '#5ef2b3' if i == 0 else '#219b77', (21+x*cell, 81+y*cell, cell-2, cell-2), border_radius=5)
            if env.food:
                x, y = env.food
                pg.draw.circle(screen, '#ff647c', (int(20+(x+.5)*cell), int(80+(y+.5)*cell)), int(cell*.32))
            title = f'{"AI" if agent else "YOU"} | SCORE {env.score} | {"GAME OVER - R to restart" if env.done else "PAUSED" if paused else "NEUROSNAKE"}'
            screen.blit(font.render(title, True, '#e7f0ff'), (20, 25))
            screen.blit(font.render('Arrows: move  Space: pause  R: reset  Esc: quit', True, '#98aec9'), (20, 650))
            pg.display.flip()
            clock.tick(args.fps)
    finally:
        pg.quit()


def main():
    parser = argparse.ArgumentParser(description='NeuroSnake: play, train or evaluate a DQN')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'play', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--backend', choices=['numpy', 'torch'], default='numpy')
        p.add_argument('--seed', type=int, default=42 if name != 'evaluate' else 10000)
        p.add_argument('--size', type=int, default=12)
        if name in ('play', 'evaluate'):
            p.add_argument('--model', required=name == 'evaluate')
        if name in ('train', 'evaluate'):
            p.add_argument('--games', type=int, default=500 if name == 'train' else 100)
            p.add_argument('--output', default='artifacts/numpy' if name == 'train' else None)
        if name == 'train':
            p.add_argument('--batch', type=int, default=64)
        if name == 'play':
            p.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    if args.size < 6 or getattr(args, 'games', 1) < 1 or getattr(args, 'batch', 1) < 1 or getattr(args, 'fps', 1) < 1:
        parser.error('size must be >=6; games, batch and fps must be positive')
    {'train': train, 'play': play, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
