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
    python3 main.py match       # 100-hand matches with persistent bankroll
    python3 main.py train       # DQN training loop (logs + learning curve)

Match logs and training output are written to `experiments/`.

## Files

- `equity.py` - hand evaluation and Monte Carlo equity estimation.
- `features.py` - the state feature extractor phi(state).
- `agents.py` - the rule-based baseline agent (equity vs pot odds).
- `match.py` - environment setup and the 100-hand match harness that carries
  chip stacks across hands and stops on bankruptcy, with CSV logging.
- `train.py` - the reinforcement-learning training loop (placeholder DQN for
  now, to be replaced by our own agent).
- `main.py` - command-line entry point that runs each mode.

## Status

Done:
- Monte Carlo equity estimator (AA preflop ~0.85, 72o ~0.35, AKs ~0.67).
- State feature extractor phi(state): equity, pot odds, stack-to-pot ratio,
  stack share, betting round, position.
- Rule-based baseline agent. Over a 2000-hand run it beats a random opponent by
  about +31 chips per hand, and wins about 64% of full 100-hand matches
  (averaged over 50 seeded, reproducible matches).
- 100-hand match harness with persistent bankroll and CSV logging.
- Training loop runs end to end and plots a learning curve.

In progress:
- Our own reinforcement-learning agent (the DQN in the loop is a placeholder).
  We start from a linear Q-learning update over phi(state), then move to a
  neural network.
- Opponent-tendency features with a Bayesian prior, added to phi(state).

## Notes

- RLCard treats one game as one hand and resets both stacks each hand, so the
  persistent bankroll and bankruptcy condition are handled by the match harness
  in `match.py`, not by RLCard itself.
- At $500 stacks with $1/$2 blinds a single all-in swings up to 500 chips, so
  individual matches are high variance. Results are reported over many matches.
