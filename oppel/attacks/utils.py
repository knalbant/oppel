from __future__ import print_function, division
import torch
import torch.nn as nn
from collections import MutableMapping
import six

class ParamDict(MutableMapping):

    def __init__(self, **kwargs):
        self.store = {}
        self._frozen_set = set()
        self.update(dict(**kwargs))

    def bulk_set(self, **kwargs):
        for key, val in kwargs.items():
            self[key] = val

    def freeze_attr(self, attr):
        if attr in self.keys():
            self._frozen_set.add(attr)
        else:
            raise AttributeError('Tried to freeze a non-existent attr: ' + attr)

    def __repr__(self):
        pre = "ParamDict("

        args = ', '.join("{}={}".format(key, val) for key, val in self.store.items())

        return pre + args + ")"

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def __delitem__(self, key):
        del self.store[key]

    def __setitem__(self, key, value):
        if key in self._frozen_set:
            raise KeyError('Attempted to modify a frozen key: {}'.format(key))

        elif not isinstance(key, six.string_types):
            raise KeyError('Attempted to use non-string value as a key')

        else:
            self.store[key] = value

    def __getitem__(self, key):
        return self.store[key]


def imgClamper(tensor):
    return tensor.clamp(min=0, max=1)


def read_synsets(path):
    """
    Reads a synset and return a dictionary of processed values

    """

    with open(path) as f:

        raw_syns = f.readlines()

        class_syns = {words[0]: ' '.join(words[1:]).split(",")
                      for words in
                      (class_line.split() for class_line in raw_syns)}

    return class_syns


def read_class_labels(path):
    """
    Reads a class label file similar toimagenet_classes.txt
    where each class is on its own line, returns an array
    of each class in the order encountered in the file.
    """

    with open(path) as f:
        raw_classes = f.readlines()

    classes = [l.strip() for l in raw_classes]

    return classes


# convenience funtions to get the highest scoring class id
def predict_u(net, x):
    return net(x.unsqueeze(0)).argmax(dim=1)

def predict(net, x):
    return net(x).argmax(dim=1)


class Normalize(nn.Module):
    """
    A pytorch module which implements feature normalization

    """
    def __init__(self, mean, std, ndims=3):
        super(Normalize, self).__init__()

        # also insert checks for mean and std being iterables

        # insert checks on this
        view_tuple = (-1,) + (1,) * (ndims - 1)

        mean_mat = torch.FloatTensor(mean).view(*view_tuple)
        std_mat = torch.FloatTensor(std).view(*view_tuple)

        self.register_buffer('mean', mean_mat)
        self.register_buffer('std',  std_mat)

    def __str__(self):
        return "Normalize(mean_size={}, std_size={})".format(self.mean.shape,
                                                             self.std.shape)

    def forward(self, x):
        return (x - self.mean) / self.std


class IdentityTransform(nn.Module):
    """
    Dummy torch.nn module that implements the identity function.

    It's primary usage in this codebase is to simplify some of
    the initalization code of certain attack classes.
    """
    def __init__(self):
        super(IdentityTransform, self).__init__()

    def forward(self, x):
        return x

    def invert_forward(self, x):
        return x
