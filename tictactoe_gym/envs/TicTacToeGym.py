import gym
import pygame
import numpy as np
from gym import spaces

class TicTacToeEnv(gym.Env):
    """
    Tic-Tac-Toe Environment that follows Open AI Gym interface
    
    """

    def __init__(self, render_mode = None, size = 3):
        """
        Initialize the environment with render mode and size.

        Parameters:
        render_mode (str): The render mode, either "human" or None.
        size (int): The size of the tic-tac-toe board. Currently only works with 3.
        """
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        #Currently will only work with default size but plan to extend to any size in future updates
        self.size = size

        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.size * self.size,), dtype = int)
        self.action_space = spaces.Discrete(size*size)
        
        self.state = np.zeros(9)
        self.current_player = 1

    def _get_obs(self):
        """
        Get the observation (size x size) from the flattened state.

        Returns:
        observation (ndarray): The observation of the environment.
        """
        return np.reshape(self.state, (-1,self.size))

    def reset(self):
        """
        Reset the environment.

        Returns:
        observation (ndarray): The observation of the environment.
        """
        self.state = np.zeros(9)
        observation = self._get_obs()
        self.current_player = 1

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def _is_game_over(self):
        """
        Check if the game is over.

        Returns:
        result (bool): True if the game is over, False otherwise.
        """
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
        """
        Get the result of the game.

        Returns:
        result (int): 1 if player 1 wins, -1 if player 2 wins, 0 if draw.
        """
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

    def _get_info(self):
        return {"info": False }

    def step(self, action):
        """
        Take a step in the environment and return the new observation, reward, if the game is terminated, and additional information.

        Parameters:
        action (int): The action to take in the environment.

        Returns:
        observation (numpy.ndarray): The observation of the environment after taking the action.
        reward (int): The reward for taking the action.
        terminated (bool): Whether the game is terminated or not.
        info (bool): Additional information, always False in this case.
        """
        
        assert self.action_space.contains(action)
        
        if self.state[action] == 0:
            self.state[action] = self.current_player

        observation = self._get_obs()

        terminated = self._is_game_over()
        reward = self._result()

        self.current_player *= -1

        if self.render_mode == "human":
            self.render()

        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def run(self, agent1, agent2, render_mode = None):
        """
        Run a game between two agents.

        Parameters:
        agent1 (Agent): The first agent to play the game.
        agent2 (Agent): The second agent to play the game.

        Returns:
        result (int): 1 if player 1 wins, -1 if player 2 wins, 0 if draw.
        """
        self.reset()
        while not self._is_game_over():
            if self.current_player == 1:
                action = agent1.get_action(self.state)
            else:
                action = agent2.get_action(self.state)
            self.step(action)

            if render_mode == "human":
                self.render()
                
        return self._result()

    def render(self, frame_rate = 0.5):
        """
        Render the current state of the environment in human-readable form.
        """
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

        self.clock.tick(frame_rate)

        self.close()

        
    def draw_grid(self):
        """
        Draw the tic-tac-toe grid on the pygame window.
        """
        bg = (255, 255, 255)
        grid = (50, 50, 50)
        self.window.fill(bg)

        for x in range(1,3):
            pygame.draw.line(self.window, grid, (0, x*100), (300, x*100), 3)
            pygame.draw.line(self.window, grid, (x*100, 0), (x*100, 300), 3)

    def draw_markers(self):
        """
        Draw the markers for the current state on the tic-tac-toe board.

        Markers for player 1 (X) will be green and markers for player 2 (O) will be red.
        """
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
        """
        Close the pygame window if it exists.
        """
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
