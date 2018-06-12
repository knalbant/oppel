from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
import numpy as np
from . import utils as ut
from . import base_attack
from collections import namedtuple


Deepfool_Return_Payload = namedtuple("Deepfool_Return_Payload", ["delta",
                                                                 "num_passes"])


def deepfool(net, inp, clamper=None, max_iter=10, overshoot=0.03, p_norm=2):
    """
    Performs the deepfool attack.

    Args:

        net (torch.nn.Module): the model to calculate an
            adversarial example for

        inp (torch.Tensor): the input to perturb
            NOTE: the dimnesion of this input should
            be one less than what the network expects
            (e.g. if the network takes minibatches of size
            10 x 3 x 32 x 32 the input the input should
            just be of dimension 3 x 32 x 32)

        clamper (callable): a callable to apply clamping
            to perturbed input features should take just a tensor
            and return the clamped tensor. Defaults to appropriate
            clamping behavior for torchvision images i.e. in the
            range of [0,1]. If no transform is desired the utility
            torch module, IdentityTransform, in the utils module can
            be passed.

        max_iter (int): the maximum number of iterations to
            run the attack for

        overshoot (float): how much to overshoot the decision
            boundary by

        p_norm (float): which p-norm to use to calculate the
            adversarial attack

    Returns:
        torch.Tensor: The computed adversarial example

        namedtuple:
    """
    if clamper is None:
        clamper = ut.imgClamper

    x_0 = inp.clone().unsqueeze(0)

    score_func = _label_score(p_norm)
    calc_r = _calculate_r(p_norm)

    # if x_0 is cloned directly this is recorded in the autograd engine
    # and messes up subsequent gradient calculations
    x_i = x_0.data.clone()
    x_i.requires_grad_(True)

    fks = net(x_i)
    ks = fks.data.numpy().flatten().argsort()[::-1]
    k_0 = ks[0]
    k_i = k_0

    num_passes = 0

    while k_i == k_0 and num_passes < max_iter:
        zero_gradients(x_i)

        p = np.inf

        fk0_grad, fk0 = _fkgrad_fk(x_i, fks, k_0)

        fk_grads = (_fkgrad_fk(x_i, fks, k) for k in ks if k != k_0)
        diffs = ((fk_grad - fk0_grad, fk - fk0) for fk_grad, fk in fk_grads)
        res = min(diffs, key=score_func)

        r = calc_r(*res)

        x_i.data += (r.data *  (1. + overshoot))

        num_passes += 1

        fks = net(x_i)
        k_i = fks.argmax(dim=1)[0].item()


    adv_ex = clamper(x_i.detach().data.clone())[0]
    r_sum = (adv_ex - inp).detach()

    payload = Deepfool_Return_Payload(r_sum, num_passes)
    return adv_ex, payload


class DeepFool(base_attack.BaseAttack):
    """
    Class implementing the DeepFool attack.

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
            to the function, deepfool, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 normalizer=None,
                 **attack_args):

        self._params = ut.ParamDict(**attack_args)

        self._attack_func = deepfool

        if normalizer is not None:
            self._model = nn.sequential(normalizer, net)
        else:
            self._model = net

    @property
    def attack_type(self):
        return base_attack.AttackType.untargeted

    @property
    def model(self):
        return self._model

    @property
    def attack_func(self):
        return self._attack_func

    @property
    def params(self):
        return self._params


### HELPER FUNCTIONS
def _fkgrad_fk(x, fks, k):
    """
    Get the input gradient and class score of a given output.
    """
    assert x.requires_grad
    zero_gradients(x)
    fks[0, k].backward(retain_graph=True)
    wk = x.grad.data.clone()

    return wk, fks[0, k]


def _label_score(p):
    """
    Helper function to determine which class boundary to push towards.

    Used as a key function for min to determine which class boundary
    to push the adversarial example towards.
    """
    if p == np.inf:
        q = 1
    else:
        q = p / (p - 1)

    def score(wk_fk):
        wk, fk = wk_fk
        return  fk.data.abs() / wk.data.norm(p=q)

    return score


def _calculate_r(p):
    """
    Closure to calculate r_i (see pg. 5 of the paper)

    Given the norm to use returns a function what
    will take w_l and f_l and return the value r.
    """
    if p == np.inf:

        def calculate(w_l, f_l):
            fl_abs = f_l.data.abs()
            w_l_1 = w_l.data.norm(p=1)
            w_l_sign = w_l.data.sign()

            return fl_abs / w_l_1 * w_l_sign

        return calculate


    else:
        q = p / (p - 1)

        def calculate(w_l, f_l):
            fl_abs = f_l.data.abs()
            w_l_norm_q = (w_l.data.norm(p=q) ** q)
            w_l_abs_q = (w_l.data.abs() ** (q - 1))

            w_l_sign = w_l.data.sign()

            return fl_abs / w_l_norm_q * w_l_abs_q * w_l_sign

        return calculate
