from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from .geometric_mean_score import gmean_scorer
from .corr_feature_reducer import CorrelationFeatureReducer
from .stratified_group_kfold import StratifiedGroupKFold


def preprocess(
    X_train, X_test, y_train, k=None, reduction_type="kbest", estimator=None
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if k is not None:
        if reduction_type == "kbest":
            selector = SelectKBest(f_classif, k=k)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        elif reduction_type == "corr":
            selector = CorrelationFeatureReducer(threshold=k)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        elif reduction_type == "pca":
            pca = PCA(n_components=k)
            X_train = pca.fit_transform(X_train, y_train)
            X_test = pca.transform(X_test)
        elif reduction_type == "rfe":
            assert estimator is not None
            rfe = RFE(estimator, n_features_to_select=k)
            X_train = rfe.fit_transform(X_train, y_train)
            X_test = rfe.transform(X_test)
    return X_train, X_test


def find_best_fold(folds, metrics_results, eval_metric="f1"):
    best_fold = None

    metrics = {
        "accuracy": 0,
        "precision": 1,
        "recall": 2,
        "specificity": 3,
        "f1": 4,
        "geometric_mean": 5,
    }

    idx_metric = metrics[eval_metric]
    idx_best_fold = np.argmax(metrics_results[:, idx_metric])
    best_fold = list(folds)[idx_best_fold]

    return best_fold, idx_best_fold


def calculate_metrics(y_true, y_pred, display=True):
    if -1 in y_pred:
        y_pred = (y_pred == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    epsilon = 1e-08

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    geometric_mean = np.sqrt(specificity * recall)

    if display:
        print(f"Accuracy: {accuracy*100:.2f}")
        print(f"Precision: {precision*100:.2f}")
        print(f"Recall: {recall*100:.2f}")
        print(f"Specificity: {specificity*100:.2f}")
        print(f"F1 Score: {f1_score*100:.2f}")
        print(f"Geometric Mean: {geometric_mean*100:.2f}")

    return accuracy, precision, recall, specificity, f1_score, geometric_mean


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")


def display_kfold_scores(other_metrics, auc_scores=None):
    others_mean = np.mean(other_metrics, axis=0)
    others_std = np.std(other_metrics, axis=0)
    print("K-Fold Results")
    if auc_scores is not None:
        print(f"AUC-ROC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Accuracy: {others_mean[0]*100:.2f} ± {others_std[0]*100:.2f}")
    print(f"Precision: {others_mean[1]*100:.2f} ± {others_std[1]*100:.2f}")
    print(f"Recall: {others_mean[2]*100:.2f} ± {others_std[2]*100:.2f}")
    print(f"Specificity: {others_mean[3]*100:.2f} ± {others_std[3]*100:.2f}")
    print(f"F1 Score: {others_mean[4]*100:.2f} ± {others_std[4]*100:.2f}")
    print(f"Geometric Mean: {others_mean[5]*100:.2f} ± {others_std[5]*100:.2f}")


def apply_grid_search(
    X, y, estimator, param_grid, scoring, cv=None, display_score=True, fit_params={}
):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        # n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y, **fit_params)

    if display_score:
        print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_params_


def apply_grid_search_grouped(
    X, y, estimator, param_grid, scoring, groups, display_score=True, fit_params={}
):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv.split(X, y, groups=groups),
        verbose=1,
    )

    grid_search.fit(X, y, **fit_params)

    if display_score:
        print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_params_


def extract_params_and_k(params, model_prefix="clf", k_key="select__k"):
    best_params = {
        k.split("__")[-1]: v for k, v in params.items() if k.startswith(model_prefix)
    }
    best_k = params[k_key]

    return best_params, best_k


def get_kfold_results(
    model,
    X,
    y,
    cv,
    best_k,
    preprocess_reduction_type="kbest",
    preprocess_estimator=None,
    sample_weights=None,
):
    folds = cv.split(X, y)

    metrics = []

    for train_idx, test_idx in folds:
        X_train_, X_test_ = X[train_idx], X[test_idx]
        y_train_, y_test_ = y[train_idx], y[test_idx]

        X_train_, X_test_ = preprocess(
            X_train_,
            X_test_,
            y_train_,
            k=best_k,
            reduction_type=preprocess_reduction_type,
            estimator=preprocess_estimator,
        )

        if sample_weights is not None:
            fit_params = {"sample_weight": sample_weights[train_idx]}
        else:
            fit_params = {}
        model.fit(X_train_, y_train_, **fit_params)

        y_pred = model.predict(X_test_)
        metrics.append(calculate_metrics(y_test_, y_pred, display=False))

    return np.array(metrics)


def get_kfold_results_grouped(
    model,
    X,
    y,
    groups,
    best_k,
    preprocess_reduction_type="kbest",
    preprocess_estimator=None,
    sample_weights=None,
):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = cv.split(X, y, groups=groups)

    metrics = []

    for train_idx, test_idx in folds:
        X_train_, X_test_ = X[train_idx], X[test_idx]
        y_train_, y_test_ = y[train_idx], y[test_idx]

        X_train_, X_test_ = preprocess(
            X_train_,
            X_test_,
            y_train_,
            k=best_k,
            reduction_type=preprocess_reduction_type,
            estimator=preprocess_estimator,
        )

        if sample_weights is not None:
            fit_params = {"sample_weight": sample_weights[train_idx]}
        else:
            fit_params = {}

        model.fit(X_train_, y_train_, **fit_params)

        y_pred = model.predict(X_test_)
        metrics.append(calculate_metrics(y_test_, y_pred, display=False))

    return np.array(metrics)
