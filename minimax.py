from connect4env import Connect4Env
import random
import numpy as np
from copy import deepcopy

def get_score(board, l):
    score = 0

    if np.any(np.abs(np.sum(board[:, :, :4], axis=2)) == l):
        score += np.sum(board[:, :, :4].flatten())
    if np.any(np.abs(np.sum(board[:, :, 1:5], axis=2)) == l):
        score += np.sum(board[:, :, 1:5].flatten())
    if np.any(np.abs(np.sum(board[:, :, 2:6], axis=2)) == l):
        score += np.sum(board[:, :, 2:6].flatten())
    if np.any(np.abs(np.sum(board[:, :, 3:7], axis=2)) == l):
        score += np.sum(board[:, :, 3:7].flatten())

    if np.any(np.abs(np.sum(board[:, :4, :], axis=1)) == l):
        score += np.sum(board[:, :4, :].flatten())
    if np.any(np.abs(np.sum(board[:, 1:5, :], axis=1)) == l):
        score += np.sum(board[:, 1:5, :].flatten())
    if np.any(np.abs(np.sum(board[:, 2:6, :], axis=1)) == l):
        score += np.sum(board[:, 2:6, :].flatten())


    # Check depth
    if np.any(np.abs(np.sum(board[:, :, :], axis=0)) == l):
        score += np.sum(board[:, :, :].flatten())

    return score

def heuristic(board, player):
    score = 0
    for l in range(2,4):
        score += get_score(board, l)
    return score

def alphabeta(player, env, depth, alpha=float("-inf"), beta=float("inf")):
    """Implementation of the alphabeta algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you need
            from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """
    def max_value(depth, env, player, is_over, alpha, beta, reward):
        if is_over or depth == 0:
            if env.check_win():
                return None, reward * 10000
            else:
                return None, heuristic(env.board, player)

        utility = float('-inf')
        best_move = None

        possible_moves = env.get_legal_moves()

        next_states = []

        for move in possible_moves:
            env_copy = deepcopy(env)
            next_state, reward, is_over, info = env_copy.step(move)
            _, next_utility = min_value(depth - 1, env_copy, player*-1, is_over, alpha, beta, reward)
            if next_utility > utility:
                utility = next_utility
                best_move = move
                alpha = max(alpha, utility)
            if utility >= beta:
                return best_move, utility

        return best_move, utility

    def min_value(depth, env, player, is_over, alpha, beta, reward):
        if is_over or depth == 0:
            if env.check_win():
                return None, reward * 10000
            else:
                return None, heuristic(env.board, player)

        utility = float('inf')
        best_move = None

        possible_moves = env.get_legal_moves()

        for move in possible_moves:
            env_copy = deepcopy(env)
            next_state, reward, is_over, info = env_copy.step(move)
            _, next_utility = max_value(depth - 1, env_copy, player*-1, is_over, alpha, beta, reward)
            if next_utility < utility:
                utility = next_utility
                best_move = move
                beta = min(beta, utility)
            if utility <= alpha:
                return best_move, utility

        return best_move, utility

    if player == 1:
        best_move, utility = max_value(depth, env, player, False, alpha, beta, 0)
    else:
        best_move, utility = min_value(depth, env, player, False, alpha, beta, 0)
    return best_move, utility