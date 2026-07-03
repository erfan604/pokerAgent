import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from features import extract_features
from agents import RuleBasedAgent
from match import make_env


def collect_data(num_hands=1000, iters=80, seed=0):
    set_seed(seed)
    env = make_env(seed=seed)
    teacher = RuleBasedAgent(iters=iters, seed=seed)
    opp = RandomAgent(num_actions=env.num_actions)
    env.set_agents([teacher, opp])
    features = []
    labels = []
    for _ in range(num_hands):
        state, player = env.reset()
        while not env.is_over():
            agent = env.agents[player]
            if player == 0:
                features.append(extract_features(state, iters=iters, seed=seed))
                action = teacher.step(state)
                labels.append(action.name)
            else:
                action = agent.step(state)
            state, player = env.step(action, agent.use_raw)
    return np.array(features), np.array(labels)


def train_classifier(num_hands=1000, iters=80, seed=0):
    X, y = collect_data(num_hands=num_hands, iters=iters, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = DecisionTreeClassifier(max_depth=6, random_state=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    labels = sorted(set(str(v) for v in y))
    print(f'collected {len(X)} decisions across {num_hands} hands')
    print(f'actions seen: {labels}')
    print(f'test accuracy: {accuracy_score(y_test, pred):.3f}')
    print(f'macro F1:      {f1_score(y_test, pred, average="macro"):.3f}')
    print('confusion matrix (rows = true, cols = predicted):')
    print(labels)
    print(confusion_matrix(y_test, pred, labels=labels))
    return clf
