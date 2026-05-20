"""
Tabular Q-learning agent trained via self-play on tictactoe_gym.

State canonicalization
----------------------
A single Q-table is used for both players by always viewing the board from the
*current* player's perspective. Multiplying the raw board by `current_player`
flips the sign of every cell so that "+1" always means "my piece" and "-1"
always means "opponent's piece". This halves the state space and lets a single
agent play either side.

Credit assignment
-----------------
Rewards in tic-tac-toe arrive only at the terminal state. We use Monte Carlo
updates: after each game, every state-action pair visited by a player is
updated toward that player's terminal reward.

Usage
-----
    python examples/q_learning.py                 # train + evaluate
    python examples/q_learning.py --episodes 100000
"""
import argparse
import random
from collections import defaultdict

import gym
import numpy as np

import tictactoe_gym  # noqa: F401  registers 'tictactoe-v0'


class TabularQLearning:
    def __init__(self, lr=0.2, epsilon=0.2):
        self.q = defaultdict(lambda: np.zeros(9))
        self.lr = lr
        self.epsilon = epsilon

    @staticmethod
    def _key(state, player):
        return tuple(int(x * player) for x in state)

    def get_action(self, state, player=1, greedy=False):
        legal = [i for i in range(9) if state[i] == 0]
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)
        q_values = self.q[self._key(state, player)]
        return max(legal, key=lambda i: q_values[i])

    def update(self, state, action, player, target):
        key = self._key(state, player)
        self.q[key][action] += self.lr * (target - self.q[key][action])


class RandomAgent:
    def get_action(self, state):
        return random.choice([i for i in range(9) if state[i] == 0])


class GreedyWrapper:
    """Adapt a Q-learning agent to the (state)->action interface env.run() expects."""

    def __init__(self, agent, player):
        self.agent = agent
        self.player = player

    def get_action(self, state):
        return self.agent.get_action(state, self.player, greedy=True)


def train(agent, episodes, report_every=5000):
    env = gym.make("tictactoe-v0").unwrapped
    window = {1: 0, 0: 0, -1: 0}

    for ep in range(1, episodes + 1):
        env.reset()
        history = []
        terminated = False

        while not terminated:
            state = env.state.copy()
            player = env.current_player
            action = agent.get_action(state, player)
            history.append((state, action, player))
            _, _, terminated, _, _ = env.step(action)

        final = env._result()
        window[final] += 1

        for s, a, p in history:
            agent.update(s, a, p, target=(final if p == 1 else -final))

        if ep % report_every == 0:
            total = sum(window.values())
            print(
                f"ep {ep:>6}: p1_wins={window[1]/total:.2%} "
                f"draws={window[0]/total:.2%} p2_wins={window[-1]/total:.2%} "
                f"states_seen={len(agent.q)}"
            )
            window = {1: 0, 0: 0, -1: 0}


def evaluate(agent, episodes=2000):
    env = gym.make("tictactoe-v0").unwrapped
    wins = draws = losses = 0

    for _ in range(episodes // 2):
        result = env.run(GreedyWrapper(agent, 1), RandomAgent())
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    for _ in range(episodes // 2):
        result = env.run(RandomAgent(), GreedyWrapper(agent, -1))
        if result < 0:
            wins += 1
        elif result > 0:
            losses += 1
        else:
            draws += 1

    return wins, draws, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--eval-games", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent = TabularQLearning()

    print(f"Training for {args.episodes} self-play episodes...")
    train(agent, args.episodes)

    print(f"\nEvaluating vs random over {args.eval_games} games...")
    wins, draws, losses = evaluate(agent, args.eval_games)
    print(
        f"  wins={wins} ({wins/args.eval_games:.1%})  "
        f"draws={draws} ({draws/args.eval_games:.1%})  "
        f"losses={losses} ({losses/args.eval_games:.1%})"
    )


if __name__ == "__main__":
    main()
