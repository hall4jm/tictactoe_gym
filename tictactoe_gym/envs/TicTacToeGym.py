import gym
import pygame
import numpy as np
from gym import spaces


ILLEGAL_MOVE_REWARD = -10


class TicTacToeEnv(gym.Env):
    """Tic-Tac-Toe environment following the OpenAI Gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None, size=3):
        if size != 3:
            raise NotImplementedError("Only size=3 is currently supported")

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.size = size

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.size * self.size,), dtype=int
        )
        self.action_space = spaces.Discrete(size * size)

        self.state = np.zeros(self.size * self.size, dtype=int)
        self.current_player = 1

    def _get_obs(self):
        return self.state.copy()

    def _legal_actions(self):
        return [i for i in range(self.size * self.size) if self.state[i] == 0]

    def _get_info(self):
        return {
            "current_player": int(self.current_player),
            "legal_actions": self._legal_actions(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.size * self.size, dtype=int)
        self.current_player = 1

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def _check_win(self):
        s = self.state
        n = self.size
        for i in range(n):
            if abs(np.sum(s[i * n:(i + 1) * n])) == n:
                return True
            if abs(np.sum(s[i::n])) == n:
                return True
        if abs(np.sum(s[::n + 1])) == n:
            return True
        if abs(np.sum(s[n - 1:-n + 1:n - 1])) == n:
            return True
        return False

    def _is_board_full(self):
        return int(np.sum(np.abs(self.state))) == self.size * self.size

    def _is_game_over(self):
        return self._check_win() or self._is_board_full()

    def _result(self):
        if self._check_win():
            return 1 if self.current_player == 1 else -1
        return 0

    def step(self, action):
        assert self.action_space.contains(action)

        if self.state[action] != 0:
            info = self._get_info()
            info["illegal_move"] = True
            return self._get_obs(), ILLEGAL_MOVE_REWARD, True, False, info

        self.state[action] = self.current_player
        terminated = self._is_game_over()
        reward = self._result()

        if not terminated:
            self.current_player *= -1

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def run(self, agent1, agent2, render_mode=None):
        """Run a full match between two agents. Returns +1/-1/0 from player 1's perspective."""
        self.reset()
        terminated = False
        while not terminated:
            agent = agent1 if self.current_player == 1 else agent2
            action = agent.get_action(self.state)
            _, _, terminated, _, _ = self.step(action)
            if render_mode == "human":
                self.render()
        return self._result()

    def render(self, frame_rate=2):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("tictactoe_gym")
            self.window = pygame.display.set_mode((300, 300))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._draw_grid()
        self._draw_markers()

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(frame_rate)

    def _draw_grid(self):
        self.window.fill((255, 255, 255))
        grid = (50, 50, 50)
        for x in range(1, 3):
            pygame.draw.line(self.window, grid, (0, x * 100), (300, x * 100), 3)
            pygame.draw.line(self.window, grid, (x * 100, 0), (x * 100, 300), 3)

    def _draw_markers(self):
        for x_pos, row in enumerate(self.state.reshape(self.size, self.size)):
            for y_pos, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.line(
                        self.window, (0, 255, 0),
                        (y_pos * 100 + 85, x_pos * 100 + 15),
                        (y_pos * 100 + 15, x_pos * 100 + 85), 10,
                    )
                    pygame.draw.line(
                        self.window, (0, 255, 0),
                        (y_pos * 100 + 15, x_pos * 100 + 15),
                        (y_pos * 100 + 85, x_pos * 100 + 85), 10,
                    )
                elif cell == -1:
                    pygame.draw.circle(
                        self.window, (255, 0, 0),
                        (y_pos * 100 + 50, x_pos * 100 + 50), 40, 10,
                    )

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
