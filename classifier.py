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


def collect_data(num_hands=1000, iters=80, seed=0, return_groups=False):
    set_seed(seed)
    env = make_env(seed=seed)
    teacher = RuleBasedAgent(iters=iters, seed=seed)
    opp = RandomAgent(num_actions=env.num_actions)
    env.set_agents([teacher, opp])
    features = []
    labels = []
    groups = []
    for hand in range(num_hands):
        state, player = env.reset()
        while not env.is_over():
            agent = env.agents[player]
            if player == 0:
                features.append(extract_features(state, iters=iters, seed=seed))
                action = teacher.step(state)
                labels.append(action.name)
                groups.append(hand)
            else:
                action = agent.step(state)
            state, player = env.step(action, agent.use_raw)
    if return_groups:
        return np.array(features), np.array(labels), np.array(groups)
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


def train_mlp_classifier(num_hands=1000, iters=80, seed=0, group_split=False,
                         hidden_layer_sizes=(32,), save=True, out_dir='experiments'):
    X, y, groups = collect_data(num_hands=num_hands, iters=iters, seed=seed, return_groups=True)
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
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
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

    validation_fraction = 0.1
    validation_count = int(np.ceil(len(y_train) * validation_fraction))
    internal_train_count = len(y_train) - validation_count
    early_stopping_enabled = (
        np.all(train_class_counts >= 2)
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
    }

    print(f'collected {len(X)} decisions across {num_hands} hands')
    print(f'training samples: {len(X_train)}')
    print(f'test samples: {len(X_test)}')
    print(f'feature count: {X.shape[1]}')
    print(f'actions seen: {labels}')
    print(f'stratified split: {can_stratify}')
    print(f'early stopping enabled: {early_stopping_enabled}')
    print(f'test accuracy: {accuracy:.3f}')
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
                        f'seed={seed}, group_split={group_split})'),
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
        },
        'warnings': [],
    }
    header = 'Milestone 2 Proof 1 - Standalone MLP Classifier\n' + '=' * 49 + '\n\n'
    with open(os.path.join(out_dir, 'mlp_results.txt'), 'w') as f:
        f.write(header + json.dumps(payload, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(loss_curve)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training loss')
    ax.set_title('Standalone MLP Training Loss')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mlp_loss_curve.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels).plot(
        cmap='Blues', ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title('Standalone MLP Test Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mlp_confusion_matrix.png'), dpi=150)
    plt.close(fig)
    print(f'wrote {out_dir}/mlp_results.txt, mlp_loss_curve.png, mlp_confusion_matrix.png')


def tune_mlp(num_hands=1000, iters=80, seed=0, group_split=True):
    X, y, groups = collect_data(num_hands=num_hands, iters=iters, seed=seed, return_groups=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
    pred = search.best_estimator_.predict(X[test_idx])
    print('best params:', search.best_params_)
    print(f'best cv macro F1: {search.best_score_:.3f}')
    print(f'held-out accuracy: {accuracy_score(y[test_idx], pred):.3f}')
    print(f'held-out macro F1: {f1_score(y[test_idx], pred, average="macro"):.3f}')
    return search
