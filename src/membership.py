import numpy as np


def determine_membership(x, abc, max, min, type="tri"):
    assert len(abc) == 3
    a, b, c = abc
    if x <= min or x >= max:
        return 1
    if type == "tri":
        if a < x and x <= b:
            return (x - a) / (b - a)
        elif b < x and x < c:
            return (c - x) / (c - b)
        else:
            return 0
    elif type == "gaussian":
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    elif type == "semicircle":
        r = c - b
        y = np.sqrt(r ** 2 - (x - b) ** 2)
        return y
    else:
        print("mf type not supported")
        assert False


def build_membership_function(universe, labels):
    num_labels = len(labels)
    assert num_labels % 2 == 1

    limits = [np.min(universe), np.max(universe)]
    universe_range = limits[1] - limits[0]
    widths = [universe_range / ((num_labels - 1) / 2.0)] * int(num_labels)
    centers = np.linspace(limits[0], limits[1], num_labels)
    abcs = [[c - w / 2, c, c + w / 2] for c, w in zip(centers, widths)]
    dic = {}
    for label, abc in zip(labels, abcs):
        dic[label] = abc
    return dic


def _trimf(x, abc):
    assert len(abc) == 3
    a, b, c = np.r_[abc]
    assert a <= b and b <= c
    y = np.zeros(len(x))

    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y
