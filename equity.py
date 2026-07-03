import random
from itertools import combinations

RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
         'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUITS = set('SHDC')


def parse(card):
    a, b = card[0], card[1]
    if a in SUITS:
        return RANKS[b], a
    return RANKS[a], b


ALL_CARDS = [parse(s + r) for s in 'SHDC' for r in '23456789TJQKA']


def rank5(cards):
    ranks = sorted((c[0] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    flush = len(set(suits)) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5
    grouped = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in grouped]
    ordered = [r for r, _ in grouped]
    if straight_high and flush:
        return (8, straight_high)
    if shape[0] == 4:
        return (7, ordered[0], ordered[1])
    if shape[0] == 3 and shape[1] == 2:
        return (6, ordered[0], ordered[1])
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if shape[0] == 3:
        return (3, *ordered)
    if shape[0] == 2 and shape[1] == 2:
        return (2, *ordered)
    if shape[0] == 2:
        return (1, *ordered)
    return (0, *ranks)


def rank7(cards):
    return max(rank5(list(c)) for c in combinations(cards, 5))


def hand_equity(hole, board=None, iters=1000, seed=None):
    board = board or []
    rng = random.Random(seed)
    hero = [parse(c) for c in hole]
    shown = [parse(c) for c in board]
    known = set(hero) | set(shown)
    deck = [c for c in ALL_CARDS if c not in known]
    need = 5 - len(shown)
    wins = ties = 0
    for _ in range(iters):
        draw = rng.sample(deck, need + 2)
        opp = draw[:2]
        full = shown + draw[2:]
        mine = rank7(hero + full)
        theirs = rank7(opp + full)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            ties += 1
    return (wins + 0.5 * ties) / iters
