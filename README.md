# Heads-up Texas Hold'em Agent (CMPT 310)

An agent that plays heads-up No-Limit Texas Hold'em. Each match is 100 hands,
$500 starting stacks, $1/$2 blinds. The agent chooses fold / check / call /
raise / all-in from its hole cards, the community cards, its bankroll, and the
opponent's past behaviour, aiming to finish the match with the most chips
without going bankrupt.

## Setup

    pip install -r requirements.txt

Built and tested with Python 3.12.

## Running

`main.py` is the entry point. Each mode runs one part of the pipeline:

    python3 main.py equity      # Monte Carlo equity sanity checks
    python3 main.py features    # feature vector for a sample state
    python3 main.py baseline    # rule-based baseline vs random (per-hand average)
    python3 main.py classifier  # decision tree that predicts the baseline's action
    python3 main.py mlp         # MLP classifier + evaluation artifacts
    python3 main.py tune        # grouped grid search over MLP hyperparameters
    python3 main.py match       # 100-hand matches with persistent bankroll
    python3 main.py opponent    # Bayesian opponent model demo + figure
    python3 main.py integrated  # opponent features appended to phi(state)
    python3 main.py qlearn      # train the RL agent, evaluate vs both opponents
    python3 main.py signif      # train, then significance test vs the baseline

Results, logs, and figures are written to `experiments/`.

`signif` reproduces the headline result end to end: it trains the RL agent
(seed selected on validation evaluation seeds; test seeds were held out of
that selection), then runs the significance test at 1000 hands and again at
4000 hands.

## Files

- `equity.py` - hand evaluation and Monte Carlo equity estimation.
- `features.py` - the state feature extractor phi(state): 9 base features plus
  6 optional opponent-tendency features (15 total when enabled).
- `agents.py` - the rule-based baseline agent (equity vs pot odds).
- `opponentModeling.py` - Beta-Binomial opponent model: VPIP, PFR, per-street
  aggression, fold-to-bet, updated from observed trajectories.
- `classifier.py` - generates labelled (features, action) data from the
  baseline playing in RLCard and trains a decision tree and an MLP. Splits are
  grouped by hand so decisions from one hand never appear in both train and
  test, and the training set is oversampled to balance rare actions.
- `qlearn.py` - our reinforcement-learning agent: a linear Q-function over
  phi(state) trained on standardized Monte Carlo returns with an
  epsilon-greedy policy, trained directly against the baseline.
- `significance.py` - the evaluation harness: mean chips/hand with normal and
  bootstrap confidence intervals, a two-sided test against zero, a power
  analysis, per-hand win/loss breakdown, and match-level results with a
  Wilson interval.
- `match.py` - environment setup and the 100-hand match harness that carries
  chip stacks across hands and stops on bankruptcy, with CSV logging.
- `train.py` - Milestone 1 DQN placeholder (RLCard's agent), kept for
  comparison; superseded by `qlearn.py`.
- `main.py` - command-line entry point that runs each mode.

## Results

- Rule-based baseline beats a random opponent by about +31 chips/hand and wins
  about 64% of 100-hand matches.
- MLP classifier: 97.9% test accuracy, 0.79 macro F1 over all five actions, on
  a hand-grouped split with a balanced training set (4000 hands of data).
- RL agent vs the baseline: +6.2 chips/hand over 4000 hands,
  95% CI [+1.8, +10.7], p = 0.006. Over 1000 hands the same agent measures
  +8.1 chips/hand but does not reach significance (p = 0.073): per-hand
  payoffs have a standard deviation of about 144 chips, so 1000 hands can
  only resolve edges above roughly 13 chips/hand.
- Match-level results stay near a coin flip (50% of 100-hand matches, Wilson
  CI 37-63%): the agent's edge comes from winning about 14% of hands at a
  mean of +185 chips while losing 86% at a mean of -22, so single matches are
  dominated by variance.

## Notes

- RLCard treats one game as one hand and resets both stacks each hand, so the
  persistent bankroll and bankruptcy condition are handled by the match harness
  in `match.py`, not by RLCard itself.
- `match.py` enforces $1/$2 blinds in `make_env()` by setting RLCard's
  `env.game.small_blind = 1` and `env.game.big_blind = 2` after environment
  creation, and keeps `game_num_players: 2` and `chips_for_each: 500`.
- At $500 stacks a single all-in swings up to 500 chips, so individual matches
  are high variance. Results are reported over many hands with confidence
  intervals.
