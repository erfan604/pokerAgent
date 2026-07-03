from equity import hand_equity


class RuleBasedAgent:
    def __init__(self, iters=200, raise_margin=0.12, shove_equity=0.85, seed=None):
        self.use_raw = True
        self.iters = iters
        self.raise_margin = raise_margin
        self.shove_equity = shove_equity
        self.seed = seed

    def step(self, state):
        obs = state['raw_obs']
        legal = {a.name: a for a in state['raw_legal_actions']}
        eq = hand_equity(obs['hand'], obs['public_cards'], iters=self.iters, seed=self.seed)
        to_call = max(obs['all_chips']) - obs['my_chips']
        pot = obs['pot']
        odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        if eq >= self.shove_equity and 'ALL_IN' in legal:
            return legal['ALL_IN']
        if eq >= odds + self.raise_margin:
            if 'RAISE_POT' in legal:
                return legal['RAISE_POT']
            if 'RAISE_HALF_POT' in legal:
                return legal['RAISE_HALF_POT']
        if to_call == 0:
            return legal.get('CHECK_CALL', legal.get('FOLD'))
        if eq >= odds:
            return legal.get('CHECK_CALL', legal.get('FOLD'))
        return legal.get('FOLD', legal.get('CHECK_CALL'))

    def eval_step(self, state):
        return self.step(state), {}
