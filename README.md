# tictactoe_gym

> A custom OpenAI Gym environment for tic-tac-toe — a clean, minimal sandbox for prototyping and testing reinforcement-learning algorithms, with a working tabular Q-learning agent that reaches **~91% win rate vs. random** after 30k self-play episodes.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gym](https://img.shields.io/badge/gym-0.26.2-0081A5.svg)
![pygame](https://img.shields.io/badge/pygame-2.1.2-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-stable-success.svg)

---

## Why this exists

Most public RL environments are either too complex to debug intuitively (Atari, MuJoCo) or too abstract to be visually satisfying (grid-world tabular tasks). Tic-tac-toe sits in a useful middle ground: it has a small discrete state space (so tabular methods like Q-learning are tractable), a clear two-player structure, and a visual board you can watch — making it ideal for sanity-checking new agents before scaling up.

This package wraps tic-tac-toe in the OpenAI Gym interface so any Gym-compatible agent can plug in without modification, plus an `env.run(agent1, agent2)` helper that handles full agent-vs-agent matches and a self-play Q-learning example that learns near-optimal play from scratch.

## What this project demonstrates

- Building a **custom Gym environment** that conforms to the `gym.Env` API (`reset`, `step`, `render`, `close`, action/observation spaces, `seed`/`options`)
- Designing a **two-player environment** with alternating turns inside a single-agent-style interface
- Implementing a **pygame renderer** for live human-watchable rollouts
- Packaging the env as an **installable Python distribution** with `setup.py` and Gym's `register()` entry point
- A working **tabular Q-learning self-play agent** with state canonicalization (same Q-table plays both sides)

---

## Installation

```bash
git clone https://github.com/hall4jm/tictactoe_gym.git
cd tictactoe_gym
pip install -e .
```

Installing in editable mode (`-e`) makes it easy to tweak the env while iterating on agents. Dependencies are pinned in `setup.py`.

> **Note on Gym vs Gymnasium.** This package targets `gym==0.26.2`, the last release before the project was forked to [Gymnasium](https://gymnasium.farama.org/). The API is broadly compatible — porting to Gymnasium is mostly a matter of swapping the import.

---

## Quickstart

```python
import gym
import tictactoe_gym  # registers the env

env = gym.make("tictactoe-v0")
obs, info = env.reset()

terminated = False
while not terminated:
    # Pick from legal actions; sampling the action space directly may hit
    # occupied cells, which now ends the episode with a -10 penalty.
    action = info["legal_actions"][0]
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Run a full match between two agents

```python
import gym
import random
import tictactoe_gym

class RandomAgent:
    def get_action(self, state):
        legal = [i for i in range(9) if state[i] == 0]
        return random.choice(legal)

env = gym.make("tictactoe-v0")
result = env.unwrapped.run(RandomAgent(), RandomAgent(), render_mode="human")
print(result)  # 1 = player 1 wins, -1 = player 2 wins, 0 = draw
```

Runnable version: [`examples/random_vs_random.py`](examples/random_vs_random.py).

### Train a Q-learning agent via self-play

```bash
python examples/q_learning.py --episodes 50000
```

A sample training run (30k episodes, eps=0.2, lr=0.2):

```
ep   5000: p1_wins=43.4% draws=36.1% p2_wins=20.5%  states_seen=3258
ep  10000: p1_wins=39.8% draws=41.0% p2_wins=19.2%  states_seen=3863
ep  20000: p1_wins=34.2% draws=49.3% p2_wins=16.5%  states_seen=4150
ep  30000: p1_wins=36.0% draws=48.9% p2_wins=15.1%  states_seen=4291

Evaluating vs random over 2000 games...
  wins=1816 (90.8%)  draws=115 (5.8%)  losses=69 (3.5%)
```

The climbing draw rate during training is the diagnostic to watch: both copies of the agent get better at not losing, so games drift toward draws (the optimal result with perfect play).

See [`examples/q_learning.py`](examples/q_learning.py) for the full implementation, including state canonicalization (one Q-table plays both sides) and Monte Carlo credit assignment.

---

## Environment specification

| | |
|--|--|
| **ID** | `tictactoe-v0` |
| **Action space** | `Discrete(9)` — cells indexed 0–8, row-major |
| **Observation space** | `Box(low=-1, high=1, shape=(9,), dtype=int)` — flat board |
| **Players** | Player 1 = `+1` ("X", plays first), Player 2 = `-1` ("O"), empty = `0` |
| **Reward** | `+1` win, `0` non-terminal or draw, `-10` for an illegal move |
| **Termination** | Three in a row, full board, or illegal move |
| **Render modes** | `"human"` (pygame window) or `None` |
| **`reset()` returns** | `(observation, info)` per Gym 0.26 |
| **`step()` returns** | `(observation, reward, terminated, truncated, info)` |
| **`info` keys** | `current_player`, `legal_actions`, and `illegal_move` on a failed step |

### Cell indexing

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

### Agent interface for `env.run()`

The `env.run(agent1, agent2)` helper expects each agent to expose:

```python
def get_action(self, state) -> int: ...
```

where `state` is the flat length-9 board (values in `{-1, 0, +1}`).

---

## Project structure

```
tictactoe_gym/
├── tictactoe_gym/
│   ├── __init__.py                 # registers 'tictactoe-v0' with gym
│   └── envs/
│       ├── __init__.py
│       └── TicTacToeGym.py         # TicTacToeEnv (gym.Env subclass)
├── examples/
│   ├── random_vs_random.py         # Two random agents play a rendered match
│   └── q_learning.py               # Tabular Q-learning self-play training + eval
├── setup.py
├── LICENSE
└── README.md
```

---

## Notes & limitations

- **Board size:** the env exposes a `size` constructor argument but only `size=3` is supported. Win-condition checks are written for the 3×3 case; extending to arbitrary `n` requires a generalized line-check.
- **Illegal-move policy:** stepping into an occupied cell ends the episode with reward `-10`. This is a strong negative signal that pushes agents toward action masking — most agents (including the Q-learning example here) only choose from `info["legal_actions"]`, so they never trigger it.
- **No external evaluation:** the Q-learning agent is benchmarked vs. a random opponent. A minimax oracle would be a stronger benchmark — see "Ideas for extension."

## Ideas for extension

- **Minimax oracle** — closed-form perfect play; useful as a benchmark to confirm the Q-learning agent has converged
- **Action masking via wrapper** — wrap `step()` so the action space dynamically excludes occupied cells
- **Arbitrary board size** — generalize the win check to `n × n`
- **DQN baseline** — overkill for 3×3 but a clean way to verify a neural-net pipeline before moving to a larger env
- **Port to [Gymnasium](https://gymnasium.farama.org/)** — modern fork of Gym; mostly a drop-in change

---

## License

MIT — see [LICENSE](LICENSE).
