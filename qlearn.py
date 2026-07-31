import os

import numpy as np

from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from features import extract_features, FEATURE_NAMES
from match import make_env, play_match, STARTING_STACK
from agents import RuleBasedAgent


ACTIONS = ['FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN']
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

EQ_EDGES = (0.4, 0.6, 0.8)
AUG_NAMES = FEATURE_NAMES + ['eq_lo', 'eq_mid', 'eq_high', 'eq_top', 'edge', 'bias']


class LinearQAgent:
    """Linear Q-function over phi(state): Q(s, a) = w_a . [phi(s), 1].

    Trained with Monte-Carlo returns (the end-of-hand payoff) and an
    epsilon-greedy behaviour policy over the legal actions.
    """

    def __init__(self, iters=80, seed=None, alpha=0.05, optimism=0.2):
        self.use_raw = True
        self.iters = iters
        self.seed = seed
        self.alpha = alpha
        self.epsilon = 0.0
        self.rng = np.random.RandomState(seed)
        self.weights = self.rng.normal(0.0, 0.01, size=(len(ACTIONS), len(AUG_NAMES)))
        self.weights[:, -1] += optimism

    def _feat(self, phi):
        buckets = np.zeros(len(EQ_EDGES) + 1)
        buckets[np.searchsorted(EQ_EDGES, phi[0])] = 1.0
        return np.concatenate([phi, buckets, [phi[0] - phi[1]], [1.0]])

    def _legal(self, state):
        legal = {a.name: a for a in state['raw_legal_actions']}
        obs = state['raw_obs']
        to_call = max(obs['all_chips']) - obs['my_chips']
        if to_call <= 0 and len(legal) > 1:
            legal.pop('FOLD', None)
        return legal

    def act(self, phi, legal, explore):
        fb = self._feat(phi)
        names = list(legal.keys())
        if explore and self.rng.rand() < self.epsilon:
            name = names[self.rng.randint(len(names))]
        else:
            q = self.weights.dot(fb)
            name = max(names, key=lambda n: q[ACTION_INDEX[n]])
        return legal[name]

    def step(self, state):
        phi = extract_features(state, iters=self.iters, seed=self.seed)
        return self.act(phi, self._legal(state), explore=True)

    def eval_step(self, state):
        phi = extract_features(state, iters=self.iters, seed=self.seed)
        return self.act(phi, self._legal(state), explore=False), {}

    def update(self, visited, ret, alpha=None):
        rate = self.alpha if alpha is None else alpha
        for phi, name in visited:
            fb = self._feat(phi)
            idx = ACTION_INDEX[name]
            q = self.weights[idx].dot(fb)
            self.weights[idx] += rate * (ret - q) * fb


def _avg_chips(agent, opponent, hands, seed):
    set_seed(seed)
    env = make_env(seed=seed)
    env.set_agents([agent, opponent])
    total = 0.0
    for _ in range(hands):
        state, player = env.reset()
        while not env.is_over():
            cur = env.agents[player]
            action = cur.eval_step(state)[0] if player == 0 else cur.step(state)
            state, player = env.step(action, cur.use_raw)
        total += env.get_payoffs()[0]
    return total / hands


def train_qlearn(episodes=3000, feat_iters=30, eval_iters=80, seed=0, alpha=0.05,
                 eps_start=0.60, eps_end=0.05, log_every=150, out_dir='experiments'):
    set_seed(seed)
    env = make_env(seed=seed)
    agent = LinearQAgent(iters=eval_iters, seed=seed, alpha=alpha)
    train_opp = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent, train_opp])

    curve = []
    window = []
    for ep in range(1, episodes + 1):
        agent.epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        state, player = env.reset()
        visited = []
        while not env.is_over():
            cur = env.agents[player]
            if player == 0:
                phi = extract_features(state, iters=feat_iters, seed=seed)
                action = agent.act(phi, agent._legal(state), explore=True)
                visited.append((phi, action.name))
            else:
                action = cur.step(state)
            state, player = env.step(action, cur.use_raw)
        payoff = env.get_payoffs()[0]
        agent.update(visited, payoff / STARTING_STACK, alpha=alpha / (1.0 + ep / 800.0))
        window.append(payoff)
        if ep % log_every == 0:
            avg = sum(window) / len(window)
            curve.append((ep, avg))
            window = []
            print(f'episode {ep:5d}  avg chips/hand (last {log_every}): {avg:+7.2f}  eps {agent.epsilon:.3f}')

    # greedy evaluation against both opponents
    avg_random = _avg_chips(agent, RandomAgent(num_actions=env.num_actions), hands=1000, seed=seed + 555)
    avg_baseline = _avg_chips(agent, RuleBasedAgent(iters=120, seed=7), hands=400, seed=seed + 777)

    nmatch = 15
    wins_random = 0
    for i in range(nmatch):
        bankrolls, _ = play_match(agent, RandomAgent(num_actions=env.num_actions), hands=100, seed=3000 + i)
        if bankrolls[0] > bankrolls[1]:
            wins_random += 1
    wins_baseline = 0
    for i in range(nmatch):
        bankrolls, _ = play_match(agent, RuleBasedAgent(iters=120, seed=1000 + i), hands=100, seed=2000 + i)
        if bankrolls[0] > bankrolls[1]:
            wins_baseline += 1
    wr_random = wins_random / nmatch
    wr_baseline = wins_baseline / nmatch

    lines = [
        'Linear Q-learning agent (Monte-Carlo returns over phi(state)), trained vs random',
        f'episodes: {episodes}   alpha: {alpha}   eps: {eps_start}->{eps_end}',
        f'greedy avg chips/hand vs random   (1000 hands): {avg_random:+.2f}',
        f'greedy avg chips/hand vs baseline  (400 hands): {avg_baseline:+.2f}',
        f'100-hand match win rate vs random   ({nmatch} matches): {wr_random:.1%}',
        f'100-hand match win rate vs baseline ({nmatch} matches): {wr_baseline:.1%}',
        '',
        'learning curve (episode, avg chips/hand vs random):',
    ]
    lines += [f'  {ep:5d}  {avg:+7.2f}' for ep, avg in curve]
    lines += ['', 'learned weights (rows = actions, cols = features + bias):',
              '  features: ' + ', '.join(AUG_NAMES)]
    for name in ACTIONS:
        w = agent.weights[ACTION_INDEX[name]]
        lines.append(f'  {name:16s} ' + ' '.join(f'{v:+.2f}' for v in w))
    report = '\n'.join(lines)
    print()
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'qlearn_results.txt'), 'w') as f:
        f.write(report + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [ep for ep, _ in curve]
    ys = [avg for _, avg in curve]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color='0.7', linewidth=1)
    ax.plot(xs, ys, marker='o')
    ax.set_xlabel('training episodes (hands)')
    ax.set_ylabel('avg chips/hand vs random (training window)')
    ax.set_title('Linear Q-learning agent: learning curve (trained vs random)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'qlearn_curve.png'), dpi=150)
    plt.close(fig)
    print(f'\nwrote {out_dir}/qlearn_results.txt and {out_dir}/qlearn_curve.png')
    return agent
