import numpy as np

FOLD = 'FOLD'
CHECK_CALL = 'CHECK_CALL'
AGGRESSIVE_ACTIONS = {'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN'}

PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
STREET_KEYS = {FLOP: 'agg_flop', TURN: 'agg_turn', RIVER: 'agg_river'}

OPPONENT_FEATURES = ['opp_vpip', 'opp_pfr', 'opp_agg_flop', 'opp_agg_turn', 
                     'opp_agg_river', 'opp_fold_to_bet']

class OpponentModel:
    def __init__(self, priorAlpha=2.0, priorBeta=2.0, decay=1.0):
        self.priorAlpha = priorAlpha
        self.priorBeta = priorBeta
        self.decay = decay
        self.stats = {
            'vpip':[priorAlpha, priorBeta],
            'pfr':[priorAlpha, priorBeta],
            'agg_flop':[priorAlpha, priorBeta],
            'agg_turn':[priorAlpha, priorBeta],
            'agg_river':[priorAlpha, priorBeta],
            'fold_to_bet':[priorAlpha, priorBeta],
        }

    def _bump(self, key, success):
        a, b = self.stats[key]
        if self.decay < 1.0:
            a = self.priorAlpha + (a - self.priorAlpha) * self.decay
            b = self.priorBeta + (b - self.priorBeta) * self.decay
        self.stats[key] = [a + (1.0 if success else 0.0),
                           b + (0.0 if success else 1.0)]

    def posteriorMean(self,key):
        a,b = self.stats[key]
        return a/(a+b)

    def features(self):
        return np.array([self.posteriorMean(k) for k in 
                         ('vpip', 'pfr', 'agg_flop', 'agg_turn','agg_river','fold_to_bet')], dtype=float)

    def update(self, obs, actName, facedRaise):
        stageIdx = obs['stage'].value if hasattr(obs['stage'], 'value') else int(obs['stage'])

        if stageIdx == PREFLOP:
            self._bump('vpip', actName != FOLD)
            self._bump('pfr', actName in AGGRESSIVE_ACTIONS)
        elif stageIdx in STREET_KEYS and (actName in AGGRESSIVE_ACTIONS or actName == CHECK_CALL):
            self._bump(STREET_KEYS[stageIdx], actName in AGGRESSIVE_ACTIONS)

        if facedRaise:
            self._bump('fold_to_bet', actName == FOLD)

def actionName(action, env):
    if hasattr(action, 'name'):
        return action.name
    return env.actions(int(action)).name

def facedRaise(obs):
    toCall = max(obs['all_chips']) - obs['my_chips']
    return toCall > 0

def updateFromTrajectory(model, trajectory, env):
    for i in range(len(trajectory) - 1):
        state, action = trajectory[i], trajectory[i + 1]
        if not isinstance(state, dict) or 'raw_obs' not in state:
            continue
        if isinstance(action, dict):
            continue
        obs = state['raw_obs']
        model.update(obs, actionName(action, env), facedRaise(obs))