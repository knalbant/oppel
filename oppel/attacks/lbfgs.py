from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools as it
import torch.nn.functional as F
from collections import namedtuple
from . import utils as ut
from . import base_attack

LBFGS_Return_Payload = namedtuple("LBFGS_Return_Payload",
                                  ["delta", "warnflag", "num_passes", "min_val"])


def lbfgs_attack(net, inp, target,
                 xmin=torch.Tensor([0]), xmax=torch.Tensor([1]),
                 c=1,
                 loss_function=F.cross_entropy,
                 optim_payload=None):

    """
    Function implementing the LBFGS-B based attack described in
    'Intriguing Properties of Neural Networks'

    NOTE: this is not quite what's desribed in the above paper as they
    describe full attack as also including a line search over the
    paramater value, c, in order to find the adversarial example with
    minimum difference to the origina input (in terms of L2 norm)

    Args:
        xmin (torch.Tensor): a tensor that should be able to have expand_as
            called on it with input tensor, inp, denoting the lower bound
            for values of inp (None should be passed if there are none).

        xmin (torch.Tensor): a tensor that should be able to have expand_as
            called on it with input tensor, inp, denoting the upper bound
            for values of inp (None should be passed if there are none).

        c (float): the relative weight to attach to the norm of the
            difference between the original input to and the currently
            computed adversarial example (see pg. 5 of the paper).

        loss (callable): the loss function to use, for things to
            work properly should take the input as the first argument, and
            label as the second.

        optim_payload (mapping, optional):
            keyword arguments that will be passed to the optimization
            function, fmin_l_bfgs_b, (fprime, args, and bounds should
            not be specified as they are handled internally in this
            function
    """
    if optim_payload is None:
        optim_payload = {}

    x = inp.unsqueeze(0)

    bounds = create_bounds_array(xmin, xmax, x)

    x_orig = x.clone().detach().numpy()
    x0 = x.clone().detach().view(-1).numpy()

    funcs = LBFGSCallableGen(net, x.shape, loss=loss_function)

    func, fprime = funcs.calculate_loss, funcs.gradient

    args = (target, x_orig, c)

    res = fmin_l_bfgs_b(func, x0,
                        fprime=fprime, bounds=bounds,
                        args=args, **optim_payload)

    ret_array, ret_payload  = res[0], res[1]

    adv = torch.Tensor(ret_array).view(inp.shape)

    delta = (inp - adv).detach()

    ret_payload = LBFGS_Return_Payload(delta,
                                       res[2]['warnflag'],
                                       res[2]['nit'],
                                       res[1])

    return adv, ret_payload


class LBFGS_Attack(base_attack.BaseAttack):
    """
    Class implementing the LBFGS-B based attack.

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        **atack_args: keyworded arguments which will be passed
            to the function, lbfgs_attack, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net, normalizer=None,
                 **attack_args):

        self._params = ut.ParamDict(**attack_args)
        self._attack_func = lbfgs_attack

        if normalizer is not None:
            self._model = nn.Sequential(normalizer, net)
        else:
            self._model = net

    @property
    def attack_type(self):
        return base_attack.AttackType.targeted_increase

    @property
    def model(self):
        return self._model

    @property
    def attack_func(self):
        return self._attack_func

    @property
    def params(self):
        return self._params


class LBFGS_Loss(nn.Module):
    """
    Module implementing the loss function for the LBFGS attack"
    """
    def __init__(self):
        super(LBFGS_Loss, self).__init__()

    def forward(self, x, x_orig, loss_val, c):
        """
        Forward function for the LBFGS Loss

        Args:
            x (torch.Tensor):
                The adversarial example being computed

            x_orig (torch.Tensor):
                The original unperturbed input

            loss_val (float)
                The value of the loss function for input x
                (e.g. the output of nn.CrossEntropyLoss()

            c (float):
                The weight assigned to the 2 norm of the
                difference between x and x_orig.
                Should be greater than 0.
        """
        assert c > 0

        norm = (x - torch.Tensor(x_orig)).norm(p=2)
        norm_val = norm * c

        return loss_val + norm_val


class LBFGSCallableGen:
    """
    Class used to create the callables that scipy's fmin_l_bfgs_b requires.

    scipy's LBFGS-B function requires two methods (if you want to use
    exact gradients), a function, func, which gives the value of
    a function evaluated at an input, and a function, fprime, which is
    the gradient (as a flattened float64 numpy array) at that input.

    This class sets things up so that the member functions, calculate_loss,
    and gradient can be used as func, and fprime, respectively.

    Args:
        model (nn.Module): the model on which the adversarial attack should
            be performed

        input_shape (iterable): an iterable of the dimensions of the input
            being attacked (should match the shape model expects)

        loss (callable, optional): the loss function to use, for things to
            work properly should take the input as the first argument, and
            label as the second.

    """

    def __init__(self, model, input_shape, loss=F.cross_entropy):
        self.model = model
        self.loss = loss

        self.y = torch.zeros(1, dtype=torch.long)
        self.input_shape = input_shape

        self.lbfgs_loss = LBFGS_Loss()

        self.x_a = None

        self.loss_val = None
        self.forward_ran = False

    def calculate_loss(self, x_flat, target, x_orig, c):
        self.forward_ran = True
        zero_gradients(self.model)
        self.y[0] = target

        x_a = self.to_original(x_flat)

        self.model(x_a)
        loss_val = self.loss(self.model(x_a), self.y)


        self.x_a = x_a
        self.loss_val = self.lbfgs_loss(x_a, x_orig, loss_val, c)

        return float(self.loss_val)

    def to_original(self, x_numpy):
        x_flat = torch.Tensor(x_numpy)
        x_flat.requires_grad = True

        return x_flat.view(self.input_shape)

    def gradient(self, x_flat, *args):
        if not self.forward_ran:
            raise RuntimeError("fmin_l_bfgs_b didn't call calculate loss before calling for the gradient")

        grad = torch.autograd.grad(self.loss_val, self.x_a)[0]

        # don't change the typecast!
        # for whatever reason scipy throws a nasty error from
        # deep within the bowels of the lbfgs optimize method
        # if the returned array is float32
        return grad.view(-1).detach().numpy().astype('float64')


def linear_bounds(x_bound, x_actual):
    """
    Function to generate an array of bounds for either the upper or lower bound.
    """
    numel = x_actual.numel()

    if x_bound is None:
        return list(it.repeat(None, numel))

    else:
        return torch.Tensor(x_bound).unsqueeze(0).expand_as(x_actual).view(-1)


def create_bounds_array(xmin, xhigh, x_actual):
    """
    Function to create the bounds array expected by fmin_l_bfgs_b

    Args:
        xmin (torch.Tensor): a tensor of lower bounds for the input
            should be 'expandable' with the original input

        xmin (torch.Tensor): a tensor of lower bounds for the input
            should be 'expandable' with the original input

        x_actual (torch.Tensor): the actual input

    Returns:
        np.array: an array of tuples containing the bounds
        (as if the tensor x_actual were flattened)

    """

    x_low = linear_bounds(xmin, x_actual)
    x_high = linear_bounds(xhigh, x_actual)

    return np.array(list(zip(x_low, x_high)))
