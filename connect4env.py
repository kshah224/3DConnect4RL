import json
import random
from collections import deque
from enum import Enum, unique
from typing import Hashable, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id='Connect4-v0',
    entry_point='Connect4Env',
    max_episode_steps=169,
)

class ResultType(Enum):
    NONE= None
    DRAW= 0
    WIN1 = 1
    WIN2 = -1
    
    def __eq__(self, other):
        return self.value == other.value

class Connect4Env(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    LOSS_REWARD = -1
    WIN_REWARD = 1
    DRAW_REWARD = 0.5
    DEF_REWARD = 0
    
    class StepResult(NamedTuple):
        result_type: ResultType
        
        def get_reward(self, player: int) -> float:
            if self.result_type is ResultType.NONE:
                return Connect4Env.DEF_REWARD
            elif self.result_type is ResultType.DRAW:
                return Connect4Env.DRAW_REWARD
            else:
                return {
                            ResultType.WIN1.value: Connect4Env.WIN_REWARD, 
                            ResultType.WIN2.value: Connect4Env.LOSS_REWARD
                        } [self.result_type.value]
        
        def is_done(self) -> bool:
            return self.result_type is not ResultType.NONE
            
    # Initialize the environment (Observation space, Action Space)
    def __init__(self, board_shape=(4, 6, 7), window_width=512, window_height=512):
        super(Connect4Env, self).__init__()
        self.board_shape = board_shape
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=board_shape, dtype=int)
        self.action_space = spaces.Discrete(board_shape[1] * board_shape[2])
        
        self.__current_player = 1
        self.__board = np.zeros(board_shape, dtype=int)
        
        self.__window_width = window_width
        self.__window_height = window_height
        
    def action_to_index(self, action:int) -> Tuple[int, int]:
        """
        Convert an action to a row, column index (row, col)

        Args:
            action (int): Action to convert

        Returns:
            Tuple[int, int]: Row, Column index
        """
        row = action // self.board_shape[2]
        col = action % self.board_shape[2]
        return row, col
    
    def index_to_action(self, row:int, col:int) -> int:
        """
        Convert a row, column index to an action

        Args:
            row (int): Row index
            col (int): Column index

        Returns:
            int: Action
        """
        return row * self.board_shape[2] + col

    def is_valid_action(self, action:int) -> bool:
        """
        Check if an action is valid

        Args:
            action (int): Action to check

        Returns:
            bool: True if the action is valid, False otherwise
        """
        row, col = self.action_to_index(action)
        return self.__board[0][row][col] == 0

    def get_valid_action_mask(self) -> np.ndarray:
        return self.__board[0] == 0

    def get_legal_moves(self) -> np.ndarray:
        action_mask = self.__board[0] == 0
        action_mask = action_mask.flatten()
        return np.where(action_mask == True)[0]

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step function to perform an action in the environment

        Args:
            action (int): Action to perform should be an integer from 0-41 (6 * 7), where 0-6 is the first row, 7-13 is the second row, and so on

        Returns:
            Tuple[np.ndarray, float, bool, dict]: Tuple containing the next state, reward, done, and info
        """
        step_result = self._step(action)
        reward = step_result.get_reward(self.__current_player)
        done = step_result.is_done()
        info = dict()
        info["valid_actions"] = self.__board[0] == 0
        
        # Switch the other player
        self.__current_player = -self.__current_player
        
        return self.__board.copy(), reward, done, info
    
    def _step(self, action:int) -> StepResult:
        result = ResultType.NONE
        
        # Check if the action is valid, if not, the other player wins
        if not self.is_valid_action(action):
            result = ResultType.WIN2 if self.__current_player == 1 else ResultType.WIN1
            print(f"Invalid Move")
            return self.StepResult(result)
        
        row, col = self.action_to_index(action)
        
        # Go from bottom to top to find the first empty cell
        for i in list(reversed(range(self.board_shape[0]))):
            if self.__board[i][row][col] == 0:
                self.__board[i][row][col] = self.__current_player
                break
        
        # Check if board is full
        if np.all(self.__board != 0):
            result = ResultType.DRAW
        else:
            # Check if the current player has won
            if self.check_win():
                result = ResultType.WIN1 if self.__current_player == 1 else ResultType.WIN2
        
        return self.StepResult(result)
    
    @property
    def board(self) -> np.ndarray:
        return self.__board.copy()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)
        info = dict()
        info["valid_actions"] = self.__board[0] == 0
        return self.__board.copy(), info
    
    def check_win(self, verbose=False) -> bool:
        """
        Check if any player has won
        """
        
        # Test Rows
        if np.any(np.abs(np.sum(self.__board[:, :, :4], axis=2)) == 4) or \
           np.any(np.abs(np.sum(self.__board[:, :, 1:5], axis=2)) == 4) or \
           np.any(np.abs(np.sum(self.__board[:, :, 2:6], axis=2)) == 4) or \
           np.any(np.abs(np.sum(self.__board[:, :, 3:7], axis=2)) == 4):
           if verbose:
                print("Row Win")
           return True
           
        # Test Columns
        if np.any(np.abs(np.sum(self.__board[:, :4, :], axis=1) == 4)) or \
           np.any(np.abs(np.sum(self.__board[:, 1:5, :], axis=1) == 4)) or \
           np.any(np.abs(np.sum(self.__board[:, 2:6, :], axis=1) == 4)):
           if verbose:
                print("Column Win")
           return True

        # Check depth
        if np.any(np.abs(np.sum(self.__board[:, :, :], axis=0)) == 4):
            if verbose:
                print("Depth Win")
            return True
        
        # Check diagonals in each 2D slice (row-wise and column-wise)
        for i in range(4):
            # Check row-col diagonals
            if np.any(np.abs(np.sum(np.diag(self.__board[i, :, :4]), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, :4], k=-1), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, :4], k=-2), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 1:5]), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 1:5], k=-1), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 1:5], k=-2), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 2:6]), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 2:6], k=-1), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 2:6], k=-2), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 3:7]), axis=0))== 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 3:7], k=-1), axis=0)) == 4) or \
               np.any(np.abs(np.sum(np.diag(self.__board[i, :, 3:7], k=-2), axis=0)) == 4):
                if verbose:
                    print("Diagonal Win")
                return True
        
        for i in range(6):
            # Check col-depth diagonals
            if np.any(np.sum(np.diag(self.__board[:, i, :4]), axis=0) == 4) or \
            np.any(np.abs(np.sum(np.diag(self.__board[:, i, 1:5]), axis=0))== 4) or \
            np.any(np.abs(np.sum(np.diag(self.__board[:, i, 2:6]), axis=0)) == 4) or \
            np.any(np.abs(np.sum(np.diag(self.__board[:, i, 3:7]), axis=0)) == 4):
                if verbose:
                    print("Diagonal Win")
                return True
        
        # Check 3D diagonals (Hard coded for now since there are only 2 valid 3d-diagonals in a 4x6x7 board)
        if np.any(np.abs(np.sum([self.__board[0, 0, 0], self.__board[1, 1, 1], self.__board[2, 2, 2], self.__board[3, 3, 3]], axis=0)) == 4) or \
           np.any(np.abs(np.sum([self.__board[0, 3, 0], self.__board[1, 2, 1], self.__board[2, 1, 2], self.__board[3, 0, 3]], axis=0)) == 4):
            if verbose:
                print("Diagonal Win")
            return True
        
        return False
               
        
