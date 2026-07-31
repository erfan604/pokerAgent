import numpy as np

from equity import hand_equity

FEATURE_NAMES = ['equity', 'pot_odds', 'spr', 'stack_share',
                 'is_preflop', 'is_flop', 'is_turn', 'is_river', 'position']

OPPONENT_FEATURE_NAMES = ['opp_vpip', 'opp_pfr', 'opp_agg_flop', 'opp_agg_turn',
                          'opp_agg_river', 'opp_fold_to_bet']


def feature_names(with_opponent=False):
    if with_opponent:
        return FEATURE_NAMES + OPPONENT_FEATURE_NAMES
    return list(FEATURE_NAMES)


def extract_features(state, position=0.5, iters=200, seed=None, opp_features=None):
    obs = state['raw_obs']
    me = obs['current_player']
    equity = hand_equity(obs['hand'], obs['public_cards'], iters=iters, seed=seed)
    to_call = max(obs['all_chips']) - obs['my_chips']
    pot = obs['pot']
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    my_stack = obs['stakes'][me]
    spr = min(my_stack / pot, 10) / 10 if pot > 0 else 0.0
    total = sum(obs['stakes'])
    stack_share = my_stack / total if total > 0 else 0.5
    stage_idx = obs['stage'].value if hasattr(obs['stage'], 'value') else int(obs['stage'])
    stage = [0, 0, 0, 0]
    if stage_idx < 4:
        stage[stage_idx] = 1
    base = np.array([equity, pot_odds, spr, stack_share, *stage, position], dtype=float)
    if opp_features is None:
        return base
    return np.concatenate([base, np.asarray(opp_features, dtype=float)])
