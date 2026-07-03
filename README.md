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

    python3 poker_agent.py equity      # Monte Carlo equity sanity checks
    python3 poker_agent.py baseline    # rule-based baseline vs random opponent
    python3 poker_agent.py train       # DQN training loop (logs + learning curve)

Training output (CSV log and learning-curve plot) is written to `experiments/`.

## Structure of poker_agent.py

- Hand evaluation + Monte Carlo equity: 5- and 7-card evaluator and
  `hand_equity()`, which estimates win probability by random rollouts of the
  unseen cards.
- Rule-based baseline agent: `RuleBasedAgent` acts on estimated equity versus
  pot odds. This is the baseline the learning agent must beat.
- Environment + training: `make_env()` sets up the RLCard heads-up No-Limit
  Hold'em environment; `train()` runs the training loop.

## Status

Done:
- Monte Carlo equity estimator (AA preflop ~0.85, 72o ~0.35, AKs ~0.67).
- Rule-based baseline agent. Over a 2000-hand match it beats a random opponent
  by about +31 chips per hand.
- Training loop runs end to end and plots a learning curve.

In progress:
- Team-built learning agent (the DQN in the loop is a placeholder for now).
- State feature extractor and the supervised classifier on real hand histories.
- Match harness that carries chip stacks across hands, since RLCard resets both
  stacks every hand and does not model a persistent bankroll on its own.
