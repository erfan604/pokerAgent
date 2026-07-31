import os
import sys

from rlcard.agents import RandomAgent
from rlcard.utils import tournament, set_seed

from equity import hand_equity
from features import extract_features, FEATURE_NAMES
from agents import RuleBasedAgent
from match import make_env, play_match
from classifier import train_classifier
from train import train
from qlearn import train_qlearn
from opponentModeling import OpponentModel, updateFromTrajectory, OPPONENT_FEATURES


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


def opponent_demo(num_hands=200, iters=80, seed=0, out_dir='experiments'):
    set_seed(seed)
    env = make_env(seed=seed)
    hero = RuleBasedAgent(iters=iters, seed=seed)
    villain = RuleBasedAgent(iters=iters, seed=seed + 1)
    env.set_agents([hero, villain])
    model = OpponentModel()
    for _ in range(num_hands):
        state, player = env.reset()
        traj = []
        while not env.is_over():
            agent = env.agents[player]
            action = agent.step(state)
            if player == 1:
                traj.append(state)
                traj.append(action)
            state, player = env.step(action, agent.use_raw)
        updateFromTrajectory(model, traj, env)

    names = list(OPPONENT_FEATURES)
    values = [float(v) for v in model.features()]
    report = '\n'.join(
        [f'opponent model after {num_hands} hands (seed {seed}):']
        + [f'  {n:16s} {v:.3f}' for n, v in zip(names, values)]
    )
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'opponent_model.txt'), 'w') as f:
        f.write(report + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(names, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel('posterior mean (Beta-Binomial)')
    ax.set_title(f'Estimated opponent tendencies ({num_hands} hands)')
    for i, v in enumerate(values):
        ax.text(min(v + 0.02, 0.95), i, f'{v:.2f}', va='center')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'opponent_model.png'), dpi=150)
    plt.close(fig)
    print(f'wrote {out_dir}/opponent_model.txt and {out_dir}/opponent_model.png')
    return model


MODES = {
    'equity': equity_checks,
    'features': feature_demo,
    'baseline': evaluate_baseline,
    'classifier': train_classifier,
    'match': run_match,
    'train': train,
    'opponent': opponent_demo,
    'qlearn': train_qlearn,
}

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'equity'
    if mode in MODES:
        MODES[mode]()
    else:
        print('usage: python3 main.py [' + ' | '.join(MODES) + ']')
