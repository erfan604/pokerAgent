import math
import os

import numpy as np

from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from features import extract_features, feature_names
from match import make_env, play_match, STARTING_STACK
from agents import RuleBasedAgent
from opponentModeling import OpponentModel, updateFromTrajectory


ACTIONS = ['FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN']
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

EQ_EDGES = (0.4, 0.6, 0.8)
EXTRA_NAMES = ['eq_lo', 'eq_mid', 'eq_high', 'eq_top', 'edge', 'bias']


def aug_names(with_opponent=False):
    return feature_names(with_opponent=with_opponent) + EXTRA_NAMES


class LinearQAgent:
    """Linear Q-function over phi(state): Q(s, a) = w_a . [phi(s), 1].

    Trained with Monte-Carlo returns (the end-of-hand payoff) and an
    epsilon-greedy behaviour policy over the legal actions. If an
    OpponentModel is attached, its six tendency estimates are appended
    to phi(state) before the equity buckets and bias.
    """

    def __init__(self, iters=80, seed=None, alpha=0.05, optimism=0.2, opp=None):
        self.use_raw = True
        self.iters = iters
        self.seed = seed
        self.alpha = alpha
        self.epsilon = 0.0
        self.opp = opp
        self.names = aug_names(with_opponent=opp is not None)
        self.rng = np.random.RandomState(seed)
        self.weights = self.rng.normal(0.0, 0.01, size=(len(ACTIONS), len(self.names)))
        self.weights[:, -1] += optimism

    def phi(self, state, iters=None):
        opp = self.opp.features() if self.opp is not None else None
        return extract_features(state, iters=iters or self.iters, seed=self.seed,
                                opp_features=opp)

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
        return self.act(self.phi(state), self._legal(state), explore=True)

    def eval_step(self, state):
        return self.act(self.phi(state), self._legal(state), explore=False), {}

    def fresh_opp(self):
        if self.opp is None:
            return None
        self.opp = OpponentModel(self.opp.priorAlpha, self.opp.priorBeta, self.opp.decay)
        return self.opp

    def update(self, visited, ret, alpha=None, baseline=0.0):
        rate = self.alpha if alpha is None else alpha
        target = ret - baseline
        for phi, name in visited:
            fb = self._feat(phi)
            idx = ACTION_INDEX[name]
            q = self.weights[idx].dot(fb)
            self.weights[idx] += rate * (target - q) * fb


def shape(ret, kind):
    if kind == 'sqrt':
        return math.copysign(math.sqrt(abs(ret)), ret)
    if kind == 'log':
        return math.copysign(math.log1p(abs(ret) * 10.0), ret)
    if kind == 'clip':
        return max(-0.2, min(0.2, ret))
    return ret


def make_trainer(env, kind, seed):
    if kind == 'baseline':
        return RuleBasedAgent(iters=120, seed=seed)
    return RandomAgent(num_actions=env.num_actions)


def _hand(env, agent, model, collect=False, feat_iters=30, seed=None):
    state, player = env.reset()
    visited = []
    traj = []
    while not env.is_over():
        cur = env.agents[player]
        if player == 0:
            if collect:
                opp = model.features() if model is not None else None
                phi = extract_features(state, iters=feat_iters, seed=seed, opp_features=opp)
                action = agent.act(phi, agent._legal(state), explore=True)
                visited.append((phi, action.name))
            else:
                action = cur.eval_step(state)[0]
        else:
            traj.append(state)
            action = cur.step(state) if collect else cur.eval_step(state)[0]
            traj.append(action)
        state, player = env.step(action, cur.use_raw)
    if model is not None:
        updateFromTrajectory(model, traj, env)
    return visited, float(env.get_payoffs()[0])


def avg_chips(agent, opponent, hands, seed):
    set_seed(seed)
    env = make_env(seed=seed)
    env.set_agents([agent, opponent])
    saved = agent.opp
    model = agent.fresh_opp()
    total = 0.0
    for _ in range(hands):
        total += _hand(env, agent, model)[1]
    agent.opp = saved
    return total / hands


def match_win_rate(agent, make_villain, nmatch, seed):
    wins = 0
    saved = agent.opp
    for i in range(nmatch):
        model = agent.fresh_opp()
        obs = None if model is None else (lambda t, e: updateFromTrajectory(model, t, e))
        bankrolls, _ = play_match(agent, make_villain(i), hands=100, seed=seed + i,
                                  observer=obs)
        if bankrolls[0] > bankrolls[1]:
            wins += 1
    agent.opp = saved
    return wins / nmatch


def train_qlearn(episodes=3000, feat_iters=None, eval_iters=80, seed=0, alpha=0.05,
                 eps_start=0.60, eps_end=0.05, log_every=150, out_dir='experiments',
                 opponent='baseline', with_opponent=True, use_baseline=True,
                 reset_every=100, normalize=True, evaluate=True, reward_shape='none'):
    set_seed(seed)
    env = make_env(seed=seed)
    feat_iters = eval_iters if feat_iters is None else feat_iters
    agent = LinearQAgent(iters=eval_iters, seed=seed, alpha=alpha,
                         opp=OpponentModel() if with_opponent else None)

    kinds = ['random', 'baseline'] if opponent == 'mix' else [opponent]
    trainers = {k: make_trainer(env, k, seed) for k in kinds}
    models = {k: OpponentModel() if with_opponent else None for k in kinds}
    seen = {k: 0 for k in kinds}

    curve = []
    window = []
    running = []
    ret_mean = 0.0
    ret_m2 = 0.0
    for ep in range(1, episodes + 1):
        agent.epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        kind = kinds[(ep - 1) % len(kinds)]
        seen[kind] += 1
        if with_opponent and seen[kind] % reset_every == 1:
            models[kind] = OpponentModel()
        model = models[kind]
        agent.opp = model
        env.set_agents([agent, trainers[kind]])
        visited, payoff = _hand(env, agent, model, collect=True,
                                feat_iters=feat_iters, seed=seed)
        ret = shape(payoff / STARTING_STACK, reward_shape)
        delta = ret - ret_mean
        ret_mean += delta / ep
        ret_m2 += delta * (ret - ret_mean)
        target = ret - (ret_mean if use_baseline else 0.0)
        if normalize and ep > 1:
            std = math.sqrt(ret_m2 / ep)
            if std > 1e-8:
                target /= std
        agent.update(visited, target, alpha=alpha / (1.0 + ep / 800.0))
        window.append(payoff)
        running.append(payoff)
        if ep % log_every == 0:
            avg = sum(window) / len(window)
            cum = sum(running) / len(running)
            curve.append((ep, avg, cum))
            window = []
            print(f'episode {ep:5d}  avg chips/hand (last {log_every}): {avg:+7.2f}  '
                  f'cumulative: {cum:+7.2f}  eps {agent.epsilon:.3f}')

    if not evaluate:
        return agent

    avg_random = avg_chips(agent, RandomAgent(num_actions=env.num_actions), 1000, seed + 555)
    avg_baseline = avg_chips(agent, RuleBasedAgent(iters=120, seed=7), 1000, seed + 777)

    nmatch = 30
    wr_random = match_win_rate(agent, lambda _: RandomAgent(num_actions=env.num_actions),
                               nmatch, 3000)
    wr_baseline = match_win_rate(agent, lambda i: RuleBasedAgent(iters=120, seed=1000 + i),
                                 nmatch, 2000)

    lines = [
        'Linear Q agent (Monte-Carlo returns over phi(state))',
        f'episodes: {episodes}   alpha: {alpha}   eps: {eps_start}->{eps_end}   '
        f'equity iters: {feat_iters} train / {eval_iters} eval',
        f'training opponent: {opponent}   opponent features: {with_opponent}   '
        f'return baseline: {use_baseline}   normalize: {normalize}   '
        f'reward shape: {reward_shape}   model reset every: {reset_every}',
        f'feature count: {len(agent.names)}',
        f'greedy avg chips/hand vs random   (1000 hands): {avg_random:+.2f}',
        f'greedy avg chips/hand vs baseline (1000 hands): {avg_baseline:+.2f}',
        f'100-hand match win rate vs random   ({nmatch} matches): {wr_random:.1%}',
        f'100-hand match win rate vs baseline ({nmatch} matches): {wr_baseline:.1%}',
        '',
        'learning curve (episode, windowed avg, cumulative avg vs training opponent):',
    ]
    lines += [f'  {ep:5d}  {avg:+7.2f}  {cum:+7.2f}' for ep, avg, cum in curve]
    lines += ['', 'learned weights (rows = actions, cols = features + bias):',
              '  features: ' + ', '.join(agent.names)]
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
    xs = [ep for ep, _, _ in curve]
    ys = [avg for _, avg, _ in curve]
    cs = [cum for _, _, cum in curve]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color='0.7', linewidth=1)
    ax.plot(xs, ys, marker='o', color='0.6', label=f'window of {log_every} hands')
    ax.plot(xs, cs, marker='o', linewidth=2, color='darkgreen', label='cumulative average')
    ax.set_xlabel('training episodes (hands)')
    ax.set_ylabel('avg chips/hand')
    ax.set_title(f'Linear Q agent: learning curve (trained vs {opponent})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'qlearn_curve.png'), dpi=150)
    plt.close(fig)
    print(f'\nwrote {out_dir}/qlearn_results.txt and {out_dir}/qlearn_curve.png')
    return agent
