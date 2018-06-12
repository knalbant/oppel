from __future__ import print_function, division
import torch
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import numpy as np
from . import utils as ut
from . import base_attack
from collections import namedtuple


MI_FGSM_Return_Payload = namedtuple("MI_FGSM_Return_Payload", ["delta"])


def mi_fgsm(net, inp, target,
            increase_score=False,
            epsilon=4./255,
            loss=None,
            mu=0.6,
            p_norm=np.inf,
            clamper=None,
            num_iter=10):
    """
    Performs either the untargeted or targeted iterative momentum
    fast gradient sign method attack.

    Args:

        net (torch.nn.Module): the model to calculate an
            adversarial example for

        inp (torch.Tensor): the input to perturb
            NOTE: the dimnesion of this input should
            be one less than what the network expects
            (e.g. if the network takes minibatches of size
            10 x 3 x 32 x 32 the input the input should
            just be of dimension 3 x 32 x 32)

        target (float or 1-d torch.Tensor): the index
            of the targeted class

        increase_score (bool): if set to false (the default
            behavior) the targeted class will have its
            score decreased, if set to true the targeted
            class will have its score increased

        mu (float): the momentum weight

        p_norm (float): the norm to bound the perturbation
            by, only support the infinity and 2 norm

        clamper (callable): a callable to apply clamping
            to perturbed input features should take just a tensor
            and return the clamped tensor. Defaults to appropriate
            clamping behavior for torchvision images i.e. in the
            range of [0,1]. If no transform is desired the utility
            torch module, IdentityTransform, in the utils module can
            be passed.

        loss (torch.nn.Module): the loss function
            to apply defaults to cross entropy loss,

        epsilon (int): the maximal perturbation to be applied
            to any one input feature

        num_iter (float): the number of iterations
            to run the attack for

    Returns:
        torch.Tensor: The computed adversarial example

        namedtuple: Currently holds a single member named
            delta which holds the difference between the
            adversarial example and original input
    """
    if p_norm == np.inf:
        p = 1
    elif p_norm == 2:
        p = 2
    else:
        raise NotImplementedError(("Iterative Momentum FGSM "
                                   "currently only supports perturbations "
                                   "bounded by the infinity or 2 norm."))

    if clamper is None:
        clamper = ut.imgClamper

    if loss is None:
        loss = nn.CrossEntropyLoss()

    target = torch.tensor(target).view(1)

    target_factor = -1 if increase_score else 1

    x_t = inp.data.unsqueeze(0).clone()
    x_t.requires_grad_(True)

    alpha = epsilon / num_iter
    g_t = 0

    for _ in range(num_iter):
        zero_gradients(x_t)

        out = net.forward(x_t)
        _loss = loss(out, target) * target_factor
        _loss.backward()

        grad_val = x_t.grad.data
        g_t = mu * g_t + grad_val / grad_val.norm(p=p)
        x_t.data = x_t.data + alpha * g_t.sign()

        x_t.data = ut.imgClamper(x_t.data)

    adv_ex = x_t.detach()[0]
    delta = adv_ex - inp
    payload = MI_FGSM_Return_Payload(delta)

    return adv_ex, payload


class MI_FGSM(base_attack.BaseAttack):
    """Abstract base class for MI-FGSM attack"""

    def __init__(self, net, normalizer=None, **attack_args):

        self._params = ut.ParamDict(**attack_args)
        self._attack_func = mi_fgsm

        if normalizer is not None:
            self._model = nn.Sequential(normalizer, net)
        else:
            self._model = nn.Sequential(net)

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

    @property
    def attack_func(self):
        return self._attack_func


class MI_FGSM_D(MI_FGSM):

    def __init__(self, net, normalizer=None, **attack_args):
        super(MI_FGSM_D, self).__init__(net, normalizer, **attack_args)

        self.params['increase_score'] = False
        self.params.freeze_attr('increase_score')

    @property
    def attack_type(self):
        return base_attack.AttackType.targeted_decrease


class MI_FGSM_I(MI_FGSM):

    def __init__(self, net, normalizer=None, **attack_args):
        super(MI_FGSM_I, self).__init__(net, normalizer, **attack_args)

        self.params['increase_score'] = False
        self.params.freeze_attr('increase_score')

    @property
    def attack_type(self):
        return base_attack.AttackType.targeted_increase
