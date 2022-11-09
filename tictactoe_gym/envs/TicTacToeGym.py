import gym
import pygame
import numpy as np
from gym import spaces

class TicTacToeEnv(gym.Env):
    """Tic-Tac-Toe Environment that follows Open AI Gym interface"""

    def __init__(self, render_mode = None, size = 3):

        #Define variables needed for rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        #Currently will only work with default size but plan to extend to any size in future updates
        self.size = size

        #Define observation and action spaces
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.size * self.size,), dtype = int)
        self.action_space = spaces.Discrete(size*size)
        
        #Initialize game
        self.state = np.zeros(9)
        self.current_player = 1

    def _get_obs(self):
        """Private function to get observation (size x size) from flattened state"""
        return np.reshape(self.state, (-1,self.size))

    def reset(self):
        self.state = np.zeros(9)
        observation = self._get_obs()
        self.current_player = 1

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def _is_game_over(self):
        #Check if game is over
        for i in range(self.size):
            if abs(np.sum(self.state[i*self.size:(i+1)*self.size])) == self.size:
                return True
            if abs(np.sum(self.state[i::self.size])) == self.size:
                return True
        #Check diagonals
        if abs(np.sum(self.state[::self.size+1])) == self.size:
            return True
        if abs(np.sum(self.state[self.size-1:-self.size+1:self.size-1])) == self.size:
            return True

        #Check if game is a draw
        if np.sum(np.abs(self.state)) == self.size*self.size:
            return True

        return False

    def _result(self):
        if self._is_game_over():
            if np.sum(np.abs(self.state)) == self.size*self.size:
                return 0
            else:
                if self.current_player == 1:
                    return 1
                else:
                    return -1
        else:
            return 0

    def step(self, action):
        assert self.action_space.contains(action)
        
        if self.state[action] == 0:
            self.state[action] = self.current_player

        observation = self._get_obs()

        terminated = self._is_game_over()
        reward = self._result()

        self.current_player *= -1

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False

    def render(self):
            if self.window is None:
                pygame.init()
                width = 300
                height = 300
                self.window = pygame.display.set_mode((width, height))
            
            if self.clock is None:
                self.clock = pygame.time.Clock()
           

            self.draw_grid()
            self.draw_markers()

            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(1)

            self.close()

        
    def draw_grid(self):
        bg = (255, 255, 255)
        grid = (50, 50, 50)
        self.window.fill(bg)

        for x in range(1,3):
            pygame.draw.line(self.window, grid, (0, x*100), (300, x*100), 3)
            pygame.draw.line(self.window, grid, (x*100, 0), (x*100, 300), 3)

    def draw_markers(self):
        x_pos = 0
        for x in self._get_obs():
            y_pos = 0
            for y in x:
                if y == 1:
                    pygame.draw.line(self.window, (0,255,0), (y_pos*100 + 85, x_pos*100 + 15), ( y_pos*100 + 15,x_pos*100 + 85),10)
                    pygame.draw.line(self.window, (0,255,0), (y_pos*100 + 15, x_pos*100 + 15), (y_pos*100 + 85,x_pos*100 + 85),10)
                if y == -1:
                    pygame.draw.circle(self.window, (255,0,0), (y_pos*100 + 50, x_pos*100 + 50), 40, 10)
                y_pos += 1        
            x_pos += 1

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
