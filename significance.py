import math
import os

import numpy as np

from rlcard.utils import set_seed

from agents import RuleBasedAgent
from match import make_env, play_match
from opponentModeling import updateFromTrajectory
from qlearn import train_qlearn


Z95 = 1.959963984540054


def hand_payoffs(agent, opponent, hands, seed):
    set_seed(seed)
    env = make_env(seed=seed)
    env.set_agents([agent, opponent])
    saved = agent.opp
    model = agent.fresh_opp()
    out = []
    for _ in range(hands):
        state, player = env.reset()
        traj = []
        while not env.is_over():
            cur = env.agents[player]
            action = cur.eval_step(state)[0]
            if player == 1:
                traj.append(state)
                traj.append(action)
            state, player = env.step(action, cur.use_raw)
        if model is not None:
            updateFromTrajectory(model, traj, env)
        out.append(float(env.get_payoffs()[0]))
    agent.opp = saved
    return np.array(out)


def bootstrap_ci(x, reps=10000, seed=0):
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(x), size=(reps, len(x)))
    means = np.sort(x[draws].mean(axis=1))
    return float(means[int(0.025 * reps)]), float(means[int(0.975 * reps)])


def wilson(wins, n):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1.0 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return centre - half, centre + half


def match_results(agent, hands, matches, seed):
    saved = agent.opp
    wins = 0
    finals = []
    for i in range(matches):
        model = agent.fresh_opp()
        obs = None if model is None else (lambda t, e: updateFromTrajectory(model, t, e))
        bankrolls, _ = play_match(agent, RuleBasedAgent(iters=120, seed=1000 + i),
                                  hands=hands, seed=seed + i, observer=obs)
        finals.append(bankrolls[0])
        if bankrolls[0] > bankrolls[1]:
            wins += 1
    agent.opp = saved
    return wins, finals


def significance(agent=None, hands=1000, matches=30, match_hands=100, seed=0,
                 out_dir='experiments', tag=''):
    if agent is None:
        agent = train_qlearn()

    payoffs = hand_payoffs(agent, RuleBasedAgent(iters=120, seed=7), hands, seed + 4242)
    n = len(payoffs)
    mean = float(payoffs.mean())
    sd = float(payoffs.std(ddof=1))
    se = sd / math.sqrt(n)
    z = mean / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    lo, hi = mean - Z95 * se, mean + Z95 * se
    blo, bhi = bootstrap_ci(payoffs, seed=seed)
    beats = bool(p < 0.05 and mean > 0)

    won = payoffs[payoffs > 0]
    lost = payoffs[payoffs < 0]
    pushed = int((payoffs == 0).sum())

    wins, finals = match_results(agent, match_hands, matches, seed + 9000)
    wr = wins / matches
    wlo, whi = wilson(wins, matches)
    busts = sum(1 for b in finals if b <= 0)

    lines = [
        f'Significance test vs rule-based baseline ({n} hands, seed {seed})',
        '',
        f'mean chips/hand:        {mean:+.3f}',
        f'sd:                     {sd:.2f}',
        f'standard error:         {se:.3f}',
        f'95% normal CI:          [{lo:+.3f}, {hi:+.3f}]',
        f'95% bootstrap CI:       [{blo:+.3f}, {bhi:+.3f}]',
        f'z:                      {z:+.3f}',
        f'two-sided p:            {p:.4g}',
        f'beats baseline at 0.05: {beats}',
        '',
        'power',
        f'  min detectable edge (0.05, 80% power): {2.8 * se:.2f} chips/hand',
        f'  hands needed for the observed {abs(mean):.2f}: '
        f'{int((2.8 * sd / mean) ** 2) if mean else 0}',
        '',
        'where the chips come from',
        f'  hands won:            {len(won)} ({len(won) / n:.1%})',
        f'  hands lost:           {len(lost)} ({len(lost) / n:.1%})',
        f'  hands pushed:         {pushed}',
        f'  mean won hand:        {won.mean():+.2f}' if len(won) else '  mean won hand:  n/a',
        f'  mean lost hand:       {lost.mean():+.2f}' if len(lost) else '  mean lost hand: n/a',
        f'  median won hand:      {np.median(won):+.2f}' if len(won) else '',
        f'  median lost hand:     {np.median(lost):+.2f}' if len(lost) else '',
        f'  largest win:          {payoffs.max():+.0f}',
        f'  largest loss:         {payoffs.min():+.0f}',
        f'  total from wins:      {won.sum():+.0f}',
        f'  total from losses:    {lost.sum():+.0f}',
        '',
        f'{match_hands}-hand matches: {matches}',
        f'match win rate:         {wr:.1%}',
        f'95% Wilson CI:          [{wlo:.1%}, {whi:.1%}]',
        f'mean final bankroll:    {np.mean(finals):.1f} (start 500)',
        f'median final bankroll:  {np.median(finals):.1f}',
        f'final bankroll range:   [{min(finals):.0f}, {max(finals):.0f}]',
        f'matches busted:         {busts}',
    ]
    report = '\n'.join(lines)
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f'significance{tag}.txt')
    png = os.path.join(out_dir, f'significance{tag}.png')
    with open(txt, 'w') as f:
        f.write(report + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cum = np.cumsum(payoffs)
    steps = np.arange(1, n + 1)
    band = Z95 * sd * np.sqrt(steps)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color='0.7', linewidth=1)
    ax.fill_between(steps, mean * steps - band, mean * steps + band, color='0.85',
                    label='95% band around fitted rate')
    ax.plot(steps, cum, color='darkgreen', label='cumulative chips')
    ax.set_xlabel('hand')
    ax.set_ylabel('cumulative chips vs baseline')
    ax.set_title(f'Agent vs baseline over {n} hands (p = {p:.3g})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'\nwrote {txt} and {png}')
    return agent, payoffs


def run_significance(seed=2, episodes=1500):
    print(f'training seed {seed}, selected on validation seeds 40000/40097/40194;')
    print('test seeds 4242 and 9000 were held out of that selection\n')
    agent = train_qlearn(episodes=episodes, seed=seed)
    significance(agent=agent, hands=1000, matches=30, tag='_1000')
    return significance(agent=agent, hands=4000, matches=50, tag='')
