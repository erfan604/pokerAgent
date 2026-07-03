import sys

from rlcard.agents import RandomAgent
from rlcard.utils import tournament, set_seed

from equity import hand_equity
from features import extract_features, FEATURE_NAMES
from agents import RuleBasedAgent
from match import make_env, play_match
from classifier import train_classifier
from train import train


def equity_checks():
    print('AA preflop     ', round(hand_equity(['SA', 'HA'], iters=3000, seed=1), 3))
    print('72o preflop    ', round(hand_equity(['H7', 'S2'], iters=3000, seed=1), 3))
    print('AKs preflop    ', round(hand_equity(['SA', 'SK'], iters=3000, seed=1), 3))
    print('AA vs board KK7', round(hand_equity(['SA', 'HA'], ['DK', 'CK', 'H7'], iters=3000, seed=1), 3))


def feature_demo():
    env = make_env()
    state, _ = env.reset()
    features = extract_features(state, position=1.0, iters=500, seed=1)
    for name, value in zip(FEATURE_NAMES, features):
        print(f'{name:12s} {round(float(value), 3)}')


def evaluate_baseline(hands=2000):
    set_seed(42)
    env = make_env(seed=42)
    env.set_agents([RuleBasedAgent(iters=150, seed=42), RandomAgent(num_actions=env.num_actions)])
    payoffs = tournament(env, hands)
    print('rule-based avg chips/hand:', round(payoffs[0], 3))
    print('random avg chips/hand:    ', round(payoffs[1], 3))


def run_match(num_matches=50, hands=100):
    env = make_env()
    wins = 0
    total_bankroll = 0.0
    for i in range(num_matches):
        agent0 = RuleBasedAgent(iters=100, seed=i)
        agent1 = RandomAgent(num_actions=env.num_actions)
        log = 'experiments/match_log.csv' if i == 0 else None
        bankrolls, _ = play_match(agent0, agent1, hands=hands, seed=i, log_path=log)
        if bankrolls[0] > bankrolls[1]:
            wins += 1
        total_bankroll += bankrolls[0]
    print(f'matches: {num_matches} x {hands} hands')
    print(f'rule-based match win rate: {wins / num_matches:.1%}')
    print(f'rule-based avg final bankroll: {total_bankroll / num_matches:.1f} (start 500)')
    print('sample match log written to experiments/match_log.csv')


MODES = {
    'equity': equity_checks,
    'features': feature_demo,
    'baseline': evaluate_baseline,
    'classifier': train_classifier,
    'match': run_match,
    'train': train,
}

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'equity'
    if mode in MODES:
        MODES[mode]()
    else:
        print('usage: python3 main.py [' + ' | '.join(MODES) + ']')
