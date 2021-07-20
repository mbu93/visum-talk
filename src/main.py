import json
import os
from collections import namedtuple
from importlib import import_module
from pathlib import Path

import click
import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from joblib import dump, load
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

fold = namedtuple("Fold", ["X", "y"])
dataset = namedtuple("Dataset", ["train", "test"])


def make_cross_folds(digits: np.ndarray, labels: np.ndarray, amount: int) -> dataset:
    skf = StratifiedKFold(n_splits=amount)
    folds = []

    for train_index, test_index in skf.split(digits, labels):
        X_train, X_test = digits[train_index], digits[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_fold = fold(X_train, y_train)
        test_fold = fold(X_test, y_test)
        folds.append(dataset(train_fold, test_fold))
    return folds


def equalize_images(images: np.ndarray) -> np.ndarray:
    mean = images.mean()
    std = images.std()
    return (images - mean) / std


@click.group()
def main():
    pass


@main.command("preprocess")
@click.option("--out", default="out", type=str)
def preprocess(out):
    out_dir = Path(out)
    digits = datasets.load_digits()
    images = equalize_images(digits.images)
    cross_folds = make_cross_folds(images, digits.target, 6)

    with open(out_dir / "folds.pickle", "wb") as fp:
        dill.dump(cross_folds, fp)

    for i in range(6):
        print("Set {}: training images shape: {}".format(i, cross_folds[i].train.X.shape))
        print("Set {}: test images shape: {}".format(i, cross_folds[i].test.X.shape))
        print("Set {}: training labels shape: {}".format(i, cross_folds[i].train.y.shape))
        print("Set {}: test labels shape: {}".format(i, cross_folds[i].test.y.shape))


@main.command("train")
@click.option("--model", type=str)
@click.option("--out", type=str)
@click.option("--args", type=str)
def train(model: str, out: str, args: str):
    keys = json.loads(args)
    clf_str = model.split(".")[-1]
    lib = model.replace(clf_str, "")[:-1]
    clf = getattr(import_module(lib), clf_str)
    clf = clf(**keys)

    with open(os.path.join(out, "folds.pickle"), "rb") as fp:
        folds = dill.load(fp)

    X, y = folds[0].train.X, folds[0].train.y
    clf.fit(X.reshape((-1, 8 * 8)), y)
    dump(clf, os.path.join(out, "clf.pickle"))
    X, y = folds[0].test.X.reshape((-1, 8 * 8)), folds[0].test.y
    score = clf.score(X, y)
    print("Accuracy: {}".format(score))
    return score


@main.command("validate")
@click.option("--out", default="/out", type=str)
def validate(out):
    score = 0.0
    cnf = np.zeros((10, 10))
    clf = load(os.path.join(out, "clf.pickle"))

    with open("{}/folds.pickle".format(out), "rb") as fp:
        folds = dill.load(fp)

    for ds in folds:
        clf.fit(ds.train.X.reshape(-1, 8 * 8), ds.train.y)
        score += clf.score(ds.test.X.reshape(-1, 8 * 8), ds.test.y)
        predictions = clf.predict(ds.test.X.reshape(-1, 8 * 8))
        cnf += confusion_matrix(ds.test.y, predictions)

    cnf /= len(cnf)
    cnf /= cnf.sum(axis=1)
    sn.heatmap(
        cnf,
        annot=True,
    )
    plt.savefig(os.path.join(out, "conf.svg"))

    cv_score = score / len(folds)
    print("Accuracy: {}".format(cv_score))
    return cv_score


if __name__ == "__main__":
    main()
