import sys
import random
from itertools import combinations

import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import tournament, Logger, reorganize, plot_curve, set_seed


# --- hand evaluation and Monte Carlo equity ---

RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
         'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUITS = set('SHDC')


def parse(card):
    a, b = card[0], card[1]
    if a in SUITS:
        return RANKS[b], a
    return RANKS[a], b


ALL_CARDS = [parse(s + r) for s in 'SHDC' for r in '23456789TJQKA']


def rank5(cards):
    ranks = sorted((c[0] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    flush = len(set(suits)) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5
    grouped = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in grouped]
    ordered = [r for r, _ in grouped]
    if straight_high and flush:
        return (8, straight_high)
    if shape[0] == 4:
        return (7, ordered[0], ordered[1])
    if shape[0] == 3 and shape[1] == 2:
        return (6, ordered[0], ordered[1])
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if shape[0] == 3:
        return (3, *ordered)
    if shape[0] == 2 and shape[1] == 2:
        return (2, *ordered)
    if shape[0] == 2:
        return (1, *ordered)
    return (0, *ranks)


def rank7(cards):
    return max(rank5(list(c)) for c in combinations(cards, 5))


def hand_equity(hole, board=None, iters=1000, seed=None):
    board = board or []
    rng = random.Random(seed)
    hero = [parse(c) for c in hole]
    shown = [parse(c) for c in board]
    known = set(hero) | set(shown)
    deck = [c for c in ALL_CARDS if c not in known]
    need = 5 - len(shown)
    wins = ties = 0
    for _ in range(iters):
        draw = rng.sample(deck, need + 2)
        opp = draw[:2]
        full = shown + draw[2:]
        mine = rank7(hero + full)
        theirs = rank7(opp + full)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            ties += 1
    return (wins + 0.5 * ties) / iters


# --- rule-based baseline agent (equity vs pot odds) ---

class RuleBasedAgent:
    def __init__(self, iters=200, raise_margin=0.12, shove_equity=0.85, seed=None):
        self.use_raw = True
        self.iters = iters
        self.raise_margin = raise_margin
        self.shove_equity = shove_equity
        self.seed = seed

    def step(self, state):
        obs = state['raw_obs']
        legal = {a.name: a for a in state['raw_legal_actions']}
        eq = hand_equity(obs['hand'], obs['public_cards'], iters=self.iters, seed=self.seed)
        to_call = max(obs['all_chips']) - obs['my_chips']
        pot = obs['pot']
        odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        if eq >= self.shove_equity and 'ALL_IN' in legal:
            return legal['ALL_IN']
        if eq >= odds + self.raise_margin:
            if 'RAISE_POT' in legal:
                return legal['RAISE_POT']
            if 'RAISE_HALF_POT' in legal:
                return legal['RAISE_HALF_POT']
        if to_call == 0:
            return legal.get('CHECK_CALL', legal.get('FOLD'))
        if eq >= odds:
            return legal.get('CHECK_CALL', legal.get('FOLD'))
        return legal.get('FOLD', legal.get('CHECK_CALL'))

    def eval_step(self, state):
        return self.step(state), {}


# --- environment and training ---

def make_env():
    config = {'game_num_players': 2, 'chips_for_each': 500, 'dealer_id': None}
    return rlcard.make('no-limit-holdem', config=config)


def make_opponent(env, kind):
    if kind == 'baseline':
        return RuleBasedAgent(iters=150, seed=42)
    return RandomAgent(num_actions=env.num_actions)


def train(episodes=1000, opponent='random'):
    env = make_env()
    poker_agent = DQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], mlp_layers=[64, 64])
    env.set_agents([poker_agent, make_opponent(env, opponent)])

    with Logger("experiments/nlhe_dqn") as logger:
        for episode in range(episodes):
            trajectories = [[] for _ in range(env.num_players)]
            state, player = env.reset()
            trajectories[player].append(state)

            while not env.is_over():
                action = env.agents[player].step(state)
                trajectories[player].append(action)
                state, player = env.step(action, env.agents[player].use_raw)
                if not env.is_over():
                    trajectories[player].append(state)

            for pid in range(env.num_players):
                trajectories[pid].append(env.get_state(pid))

            payoffs = env.get_payoffs()
            trajectories = reorganize(trajectories, payoffs)

            for transition in trajectories[0]:
                poker_agent.feed(transition)

            if episode % 50 == 0:
                logger.log_performance(env.timestep, tournament(env, 1000)[0])

        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, "poker_agent")
    return poker_agent


def evaluate_baseline(hands=2000):
    set_seed(42)
    env = make_env()
    env.set_agents([RuleBasedAgent(iters=150, seed=42), RandomAgent(num_actions=env.num_actions)])
    payoffs = tournament(env, hands)
    print('rule-based avg chips/hand:', round(payoffs[0], 3))
    print('random avg chips/hand:    ', round(payoffs[1], 3))


def equity_checks():
    print('AA preflop     ', round(hand_equity(['SA', 'HA'], iters=3000, seed=1), 3))
    print('72o preflop    ', round(hand_equity(['H7', 'S2'], iters=3000, seed=1), 3))
    print('AKs preflop    ', round(hand_equity(['SA', 'SK'], iters=3000, seed=1), 3))
    print('AA vs board KK7', round(hand_equity(['SA', 'HA'], ['DK', 'CK', 'H7'], iters=3000, seed=1), 3))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'equity'
    if mode == 'equity':
        equity_checks()
    elif mode == 'baseline':
        evaluate_baseline()
    elif mode == 'train':
        train(episodes=1000, opponent='random')
