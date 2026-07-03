import os
import csv

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from agents import RuleBasedAgent


def make_env(seed=None):
    config = {'game_num_players': 2, 'chips_for_each': 500, 'dealer_id': None, 'seed': seed}
    return rlcard.make('no-limit-holdem', config=config)


def make_opponent(env, kind):
    if kind == 'baseline':
        return RuleBasedAgent(iters=150, seed=42)
    return RandomAgent(num_actions=env.num_actions)


def play_match(agent0, agent1, hands=100, start_bankroll=500, log_path=None, seed=None):
    if seed is not None:
        set_seed(seed)
    env = make_env(seed=seed)
    env.set_agents([agent0, agent1])
    bankrolls = [float(start_bankroll), float(start_bankroll)]
    rows = []
    for h in range(1, hands + 1):
        _, payoffs = env.run(is_training=False)
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
