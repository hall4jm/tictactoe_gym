"""Play a single tic-tac-toe game between two random-policy agents."""
import gym
import random

import tictactoe_gym  # noqa: F401  registers 'tictactoe-v0'


class RandomAgent:
    def get_action(self, state):
        legal = [i for i in range(9) if state[i] == 0]
        return random.choice(legal)


if __name__ == "__main__":
    env = gym.make("tictactoe-v0")
    env.reset()
    result = env.run(RandomAgent(), RandomAgent(), render_mode="human")
    print({1: "player 1 wins", -1: "player 2 wins", 0: "draw"}[result])
    env.close()
