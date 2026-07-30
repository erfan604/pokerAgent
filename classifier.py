import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def train_mlp_classifier(num_hands=1000, iters=80, seed=0):
    X, y = collect_data(num_hands=num_hands, iters=iters, seed=seed)
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
            hidden_layer_sizes=(32,),
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

    return model, results
