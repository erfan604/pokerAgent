import json
import os
import sys
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rlcard.agents import RandomAgent
from rlcard.utils import set_seed

from features import extract_features
from agents import RuleBasedAgent
from match import make_env
from opponentModeling import OpponentModel, updateFromTrajectory


def collect_data(num_hands=1000, iters=80, seed=0, return_groups=False, with_opponent=True):
    set_seed(seed)
    env = make_env(seed=seed)
    teacher = RuleBasedAgent(iters=iters, seed=seed)
    opp = RandomAgent(num_actions=env.num_actions)
    env.set_agents([teacher, opp])
    model = OpponentModel() if with_opponent else None
    features = []
    labels = []
    groups = []
    for hand in range(num_hands):
        state, player = env.reset()
        traj = []
        while not env.is_over():
            agent = env.agents[player]
            if player == 0:
                opp_feats = model.features() if model is not None else None
                features.append(extract_features(state, iters=iters, seed=seed,
                                                 opp_features=opp_feats))
                action = teacher.step(state)
                labels.append(action.name)
                groups.append(hand)
            else:
                traj.append(state)
                action = agent.step(state)
                traj.append(action)
            state, player = env.step(action, agent.use_raw)
        if model is not None:
            updateFromTrajectory(model, traj, env)
    if return_groups:
        return np.array(features), np.array(labels), np.array(groups)
    return np.array(features), np.array(labels)


def grouped_split(X, y, groups, classes, test_size=0.2, seed=0, tries=10):
    splitter = GroupShuffleSplit(n_splits=tries, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if len(np.setdiff1d(classes, np.unique(y[train_idx]))) == 0:
            return train_idx, test_idx
    raise ValueError('no grouped split keeps every action class in the training set')


def balanced_index(y, seed):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    picks = []
    for value in classes:
        pool = np.flatnonzero(y == value)
        picks.append(pool)
        short = target - len(pool)
        if short > 0:
            picks.append(rng.choice(pool, short, replace=True))
    order = np.concatenate(picks)
    rng.shuffle(order)
    return order


def train_classifier(num_hands=1000, iters=80, seed=0, with_opponent=True):
    X, y, groups = collect_data(num_hands=num_hands, iters=iters, seed=seed,
                                return_groups=True, with_opponent=with_opponent)
    classes = np.unique(y)
    train_idx, test_idx = grouped_split(X, y, groups, classes, seed=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = DecisionTreeClassifier(max_depth=6, random_state=seed, class_weight='balanced')
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


def train_mlp_classifier(num_hands=4000, iters=80, seed=0, group_split=True,
                         hidden_layer_sizes=(32,), save=True, out_dir='experiments',
                         with_opponent=True, balance=True):
    X, y, groups = collect_data(num_hands=num_hands, iters=iters, seed=seed,
                                return_groups=True, with_opponent=with_opponent)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError('MLP training requires at least two feature rows')

    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError('MLP training requires at least two action classes')

    test_size = 0.2
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    can_stratify = (
        np.all(class_counts >= 2)
        and test_count >= len(classes)
        and train_count >= len(classes)
    )
    stratify = y if can_stratify else None
    if group_split:
        train_idx, test_idx = grouped_split(X, y, groups, classes,
                                            test_size=test_size, seed=seed)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        can_stratify = False
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

    train_classes, train_class_counts = np.unique(y_train, return_counts=True)
    missing_classes = np.setdiff1d(classes, train_classes)
    if len(missing_classes) > 0:
        missing = ', '.join(str(value) for value in missing_classes)
        raise ValueError(f'Training split is missing action classes: {missing}')

    if balance:
        order = balanced_index(y_train, seed)
        X_train, y_train = X_train[order], y_train[order]
        train_classes, train_class_counts = np.unique(y_train, return_counts=True)

    validation_fraction = 0.1
    validation_count = int(np.ceil(len(y_train) * validation_fraction))
    internal_train_count = len(y_train) - validation_count
    early_stopping_enabled = bool(
        not balance
        and np.all(train_class_counts >= 2)
        and validation_count >= len(train_classes)
        and internal_train_count >= len(train_classes)
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=early_stopping_enabled,
            validation_fraction=validation_fraction,
            n_iter_no_change=20,
            random_state=seed,
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mlp = model.named_steps['mlp']
    labels = [str(value) for value in classes]
    accuracy = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred, labels=labels)
    report_text = classification_report(
        y_test,
        pred,
        labels=labels,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    loss_curve = [float(value) for value in mlp.loss_curve_]
    raw_validation_scores = getattr(mlp, 'validation_scores_', None)
    validation_scores = (
        None
        if raw_validation_scores is None
        else [float(value) for value in raw_validation_scores]
    )

    results = {
        'sample_count': int(len(X)),
        'training_sample_count': int(len(X_train)),
        'test_sample_count': int(len(X_test)),
        'feature_count': int(X.shape[1]),
        'action_classes': labels,
        'stratified_split': bool(can_stratify),
        'early_stopping_enabled': bool(early_stopping_enabled),
        'test_accuracy': float(accuracy),
        'confusion_matrix': matrix.tolist(),
        'classification_report': report,
        'training_iterations': int(mlp.n_iter_),
        'loss_curve': loss_curve,
        'validation_scores': validation_scores,
        'macro_f1': float(f1_score(y_test, pred, average='macro')),
        'macro_f1_tested': float(f1_score(y_test, pred, average='macro',
                                          labels=np.unique(y_test), zero_division=0)),
        'opponent_features': bool(with_opponent),
        'balanced_training_set': bool(balance),
        'test_class_distribution': {str(c): int(n) for c, n in
                                    zip(*np.unique(y_test, return_counts=True))},
    }

    print(f'collected {len(X)} decisions across {num_hands} hands')
    print(f'training samples: {len(X_train)}')
    print(f'test samples: {len(X_test)}')
    print(f'feature count: {X.shape[1]}')
    print(f'actions seen: {labels}')
    print(f'stratified split: {can_stratify}')
    print(f'early stopping enabled: {early_stopping_enabled}')
    print(f'grouped split by hand: {group_split}')
    print(f'opponent features: {with_opponent}')
    print(f'balanced training set: {balance}')
    print(f'test accuracy: {accuracy:.3f}')
    print(f'macro F1 (all {len(classes)} classes): {results["macro_f1"]:.3f}')
    print(f'macro F1 (classes present in test): {results["macro_f1_tested"]:.3f}')
    print(f'test class distribution: {results["test_class_distribution"]}')
    print('confusion matrix (rows = true, cols = predicted):')
    print(labels)
    print(matrix)
    print('classification report:')
    print(report_text)
    print(f'training iterations: {mlp.n_iter_}')
    print(f'loss curve: {loss_curve}')
    if validation_scores is not None:
        print(f'validation scores: {validation_scores}')

    if save:
        _save_artifacts(results, matrix, labels, loss_curve, validation_scores, y, out_dir,
                        num_hands, iters, seed, hidden_layer_sizes, group_split,
                        validation_fraction)

    return model, results


def _save_artifacts(results, matrix, labels, loss_curve, validation_scores, y, out_dir,
                    num_hands, iters, seed, hidden_layer_sizes, group_split,
                    validation_fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    classes, counts = np.unique(y, return_counts=True)
    steps = np.diff(loss_curve)
    payload = {
        'configuration': {
            'command': (f'train_mlp_classifier(num_hands={num_hands}, iters={iters}, '
                        f'seed={seed}, group_split={group_split}, '
                        f'with_opponent={results["opponent_features"]}, '
                        f'balance={results["balanced_training_set"]})'),
            'seed': seed,
            'num_hands': num_hands,
            'equity_iterations': iters,
            'pipeline': ['StandardScaler', 'MLPClassifier'],
            'hidden_layer_sizes': list(hidden_layer_sizes),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'early_stopping': results['early_stopping_enabled'],
            'validation_fraction': validation_fraction,
            'n_iter_no_change': 20,
            'random_state': seed,
            'group_split_by_hand': group_split,
            'opponent_features': results['opponent_features'],
            'balanced_training_set': results['balanced_training_set'],
        },
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'scikit_learn': sklearn.__version__,
            'matplotlib': matplotlib.__version__,
        },
        'dataset': {
            'sample_count': results['sample_count'],
            'feature_count': results['feature_count'],
            'class_distribution': {str(c): int(n) for c, n in zip(classes, counts)},
            'training_sample_count': results['training_sample_count'],
            'test_sample_count': results['test_sample_count'],
        },
        'evaluation': {
            'action_classes': results['action_classes'],
            'stratified_split': results['stratified_split'],
            'early_stopping_enabled': results['early_stopping_enabled'],
            'test_accuracy': results['test_accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'classification_report': results['classification_report'],
            'training_iterations': results['training_iterations'],
            'first_loss': loss_curve[0],
            'final_loss': loss_curve[-1],
            'loss_generally_decreased': bool(loss_curve[-1] < loss_curve[0]),
            'decreasing_step_fraction': float(np.mean(steps < 0)) if len(steps) else 0.0,
            'loss_curve': loss_curve,
            'validation_scores': validation_scores,
            'macro_f1': results['macro_f1'],
            'macro_f1_tested': results['macro_f1_tested'],
            'test_class_distribution': results['test_class_distribution'],
        },
        'warnings': [],
    }
    header = 'Standalone MLP Classifier\n' + '=' * 25 + '\n\n'
    with open(os.path.join(out_dir, 'mlp_results.txt'), 'w') as f:
        f.write(header + json.dumps(payload, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(loss_curve, color='darkgreen')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training loss')
    ax.set_title('Standalone MLP Training Loss')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mlp_loss_curve.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        cmap='Greens', ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title('Standalone MLP Test Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mlp_confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f'wrote {out_dir}/mlp_results.txt, mlp_loss_curve.png, mlp_confusion_matrix.png')


def tune_mlp(num_hands=1000, iters=80, seed=0, with_opponent=True, balance=True):
    X, y, groups = collect_data(num_hands=num_hands, iters=iters, seed=seed,
                                return_groups=True, with_opponent=with_opponent)
    classes = np.unique(y)
    train_idx, test_idx = grouped_split(X, y, groups, classes, seed=seed)
    grid = {
        'mlp__hidden_layer_sizes': [(16,), (32,), (64,), (32, 16)],
        'mlp__alpha': [1e-4, 1e-3, 1e-2],
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=seed)),
    ])
    inner = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=inner, scoring='f1_macro', n_jobs=-1)
    X_fit, y_fit, g_fit = X[train_idx], y[train_idx], groups[train_idx]
    if balance:
        order = balanced_index(y_fit, seed)
        X_fit, y_fit, g_fit = X_fit[order], y_fit[order], g_fit[order]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        search.fit(X_fit, y_fit, groups=g_fit)
    pred = search.best_estimator_.predict(X[test_idx])
    missing = np.setdiff1d(classes, np.unique(y[test_idx]))
    print('best params:', search.best_params_)
    print(f'best cv macro F1: {search.best_score_:.3f}')
    print(f'held-out accuracy: {accuracy_score(y[test_idx], pred):.3f}')
    print(f'held-out macro F1: {f1_score(y[test_idx], pred, average="macro"):.3f}')
    print(f'classes absent from the held-out split: {list(missing)}')
    return search
