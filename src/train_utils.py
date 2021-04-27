import argparse

import numpy as np
from tqdm import tqdm

from .membership import build_membership_function
from .nefclass import NefClassModel


seed = 8


def train(args, labels, train_data, train_targets, test_data, test_targets, universe_max, universe_min, verbose=False):
    model = NefClassModel(
        num_input_units=args.num_input_units,
        num_fuzzy_sets=args.num_sets,
        kmax=args.kmax,
        output_units=args.output_units,
        universe_max=universe_max,
        universe_min=universe_min,
        membership_type=args.mf,
    )
    abcs = [build_membership_function(train_data[d], labels) for d in range(train_data.shape[1])]
    model.init_fuzzy_sets(abcs)

    for i, (r, t) in enumerate(zip(train_data, train_targets)):
        model.learn_rule(r, t)

    if verbose:
        print(f"model learnt {model.get_num_rules()} rules")

    """check accuracy after rule learning"""
    if verbose:
        print(f"Accuracy on training set after rule learning: {check_accuracy(model, train_data, train_targets):.2f}%")

    """fuzzy set learning"""
    if verbose:
        print("==== start fuzzy set learning ====")
    best_acc_epoch_pair = [-1, -1]
    test_accs = []
    for e in tqdm(range(args.num_epoch)):
        for i, (r, t) in enumerate(zip(train_data, train_targets)):
            output = model(r, t)
            delta = [1 - output[i] if i == t else 0 - output[i] for i in range(len(output))]
            model.update_fuzzy_sets(args.sigma, delta)
        epoch_acc = check_accuracy(model, train_data, train_targets)
        test_accs.append(check_accuracy(model, test_data, test_targets))
        if epoch_acc > best_acc_epoch_pair[0]:
            best_acc_epoch_pair[0] = epoch_acc
            best_acc_epoch_pair[1] = e
        # early stopping
        if e - best_acc_epoch_pair[1] > args.num_epoch / 10:
            break
        if e % 5 == 0:
            if verbose:
                print(f"Epoch {e}: {epoch_acc:.2f}%")

    if verbose:
        print(f"Best accuracy {best_acc_epoch_pair[0]:.2f}% at epoch {best_acc_epoch_pair[1]}")
    if verbose:
        print(f"Accuracy on test set after fuzzy set learning: {test_accs[best_acc_epoch_pair[1]]:.2f}%")
    return best_acc_epoch_pair[0], test_accs[best_acc_epoch_pair[1]]


def check_accuracy(model, data, targets):
    correct = 0
    total = 0
    for i, (r, t) in enumerate(zip(data, targets)):
        # print(i)
        output = model(r, t)
        pred_class = output.index(max(output))
        if pred_class == t:
            correct += 1
        total += 1

    return 100 * correct / total


def main(args):

    """
    load dataset
    """
    if args.dataset == "iris":
        terms = load_iris(args)
    elif args.dataset == "bc":
        terms = load_breast_cancer(args)
    elif args.dataset == "wbc":
        terms = load_breast_cancer_wisconsin(args)
    elif args.dataset == "wine":
        terms = load_wine(args)
    else:
        print("dataset does not exist")
        assert False
    """
    define linguistic variable
    """
    if args.num_sets == 5:
        labels = ["lower", "low", "average", "high", "higher"]
    elif args.num_sets == 3:
        labels = ["low", "average", "high"]
    elif args.num_sets == 7:
        labels = ["lowest", "lower", "low", "average", "high", "higher", "highest"]
    elif args.num_sets == 9:
        labels = ["extremely low", "lowest", "lower", "low", "average", "high", "higher", "highest", "extremely high"]
    else:
        print("only 3/5/7 sets supported")
        assert False
    if args.cv:
        from sklearn.model_selection import KFold

        data, targets, universe_max, universe_min = terms
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=seed)
        kf.get_n_splits(data)
        cv_train_acc, cv_test_acc = [], []
        for train_idxs, test_idxs in kf.split(data):
            train_data, train_targets, test_data, test_targets = (
                data[train_idxs],
                targets[train_idxs],
                data[test_idxs],
                targets[test_idxs],
            )
            train_acc, test_acc = train(
                args,
                labels,
                train_data,
                train_targets,
                test_data,
                test_targets,
                universe_max,
                universe_min,
                verbose=args.v,
            )
            cv_train_acc.append(train_acc)
            cv_test_acc.append(test_acc)
        print(
            f"Sigma:{args.sigma}, kmax:{args.kmax}, num_sets:{args.num_sets}, Train: {np.mean(train_acc):.2f}% Test: {np.mean(test_acc):.2f}%"
        )
    else:
        train_data, train_targets, test_data, test_targets, universe_max, universe_min = terms
        train(
            args,
            labels,
            train_data,
            train_targets,
            test_data,
            test_targets,
            universe_max,
            universe_min,
            verbose=args.v,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFCLASS")
    parser.add_argument("--dataset", default="iris", type=str, help="dataset to load")
    parser.add_argument("--sigma", default=0.01, type=float, help="learning rate")
    parser.add_argument("--num_epoch", default=500, type=int, help="number of epoch for fuzzy set learning")
    parser.add_argument("--num_sets", default=5, type=int, help="number of fuzzy sets")
    parser.add_argument("--kmax", default=100, type=int, help="maximum number of rules")
    parser.add_argument(
        "--rule_learning",
        default="original",
        type=str,
        help="rule learning method to use. Default is the original implementation. Use any other strings as input to perform best per class.",
    )
    parser.add_argument("--cv", default=False, action="store_true", help="do 10 fold cross validation?")
    parser.add_argument("--kfold", default=10, type=int, help="number of k fold")
    parser.add_argument("-v", default=False, action="store_true", help="verbosity")
    parser.add_argument(
        "--mf", default="tri", type=str, help="membership function to use. Default: tri. Options: gaussian, semicircle"
    )

    args = parser.parse_args()
    main(args)
