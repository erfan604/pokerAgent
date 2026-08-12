import os
import csv

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from agents import RuleBasedAgent


NUM_PLAYERS = 2
STARTING_STACK = 500
SMALL_BLIND = 1
BIG_BLIND = 2


def _enforce_blinds(env):
    game = getattr(env, 'game', None)
    if game is None or not hasattr(game, 'small_blind') or not hasattr(game, 'big_blind'):
        raise RuntimeError('RLCard no-limit-holdem game does not expose blind attributes')
    game.small_blind = SMALL_BLIND
    game.big_blind = BIG_BLIND
    if game.small_blind != SMALL_BLIND or game.big_blind != BIG_BLIND:
        raise RuntimeError('Failed to enforce small blind 1 and big blind 2 in RLCard environment')


def make_env(seed=None):
    config = {
        'game_num_players': NUM_PLAYERS,
        'chips_for_each': STARTING_STACK,
        'dealer_id': None,
        'seed': seed,
    }
    env = rlcard.make('no-limit-holdem', config=config)
    _enforce_blinds(env)
    return env


def make_opponent(env, kind):
    if kind == 'baseline':
        return RuleBasedAgent(iters=150, seed=42)
    return RandomAgent(num_actions=env.num_actions)


def play_match(agent0, agent1, hands=100, start_bankroll=500, log_path=None, seed=None,
               observer=None):
    if seed is not None:
        set_seed(seed)
    env = make_env(seed=seed)
    env.set_agents([agent0, agent1])
    bankrolls = [float(start_bankroll), float(start_bankroll)]
    rows = []
    for h in range(1, hands + 1):
        state, player = env.reset()
        traj = []
        while not env.is_over():
            cur = env.agents[player]
            action = cur.eval_step(state)[0]
            if player == 1:
                traj.append(state)
                traj.append(action)
            state, player = env.step(action, cur.use_raw)
        payoffs = env.get_payoffs()
        if observer is not None:
            observer(traj, env)
        bankrolls[0] += payoffs[0]
        bankrolls[1] += payoffs[1]
        rows.append((h, round(float(payoffs[0]), 2), round(float(payoffs[1]), 2),
                     round(bankrolls[0], 2), round(bankrolls[1], 2)))
        if bankrolls[0] <= 0 or bankrolls[1] <= 0:
            break
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hand', 'payoff_0', 'payoff_1', 'bankroll_0', 'bankroll_1'])
            writer.writerows(rows)
    return bankrolls, rows
