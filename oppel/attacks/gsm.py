from __future__ import print_function, division
import torch
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
from . import utils as ut
from . import base_attack
from collections import namedtuple


IGSM_Return_Payload = namedtuple("IGSM_Return_Payload", ["delta"])


def untargeted_igsm(net, inp,
                    loss=None,
                    clamper=None,
                    epsilon=4./255,
                    num_iter=20):
    """
    Performs an iterated gradient sign method attack.

    Args:

        net (torch.nn.Module): the model to calculate an
            adversarial example for

        inp (torch.Tensor): the input to perturb
            NOTE: the dimnesion of this input should
            be one less than what the network expects
            (e.g. if the network takes minibatches of size
            10 x 3 x 32 x 32 the input the input should
            just be of dimension 3 x 32 x 32)

        loss (torch.nn.Module): the loss function
            to apply defaults to cross entropy loss,

        clamper (callable): a callable to apply clamping
            to perturbed input features should take just a tensor
            and return the clamped tensor. Defaults to appropriate
            clamping behavior for torchvision images i.e. in the
            range of [0,1]. If no transform is desired the utility
			torch module, IdentityTransform, in the utils module can
			be passed.

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

    if loss is None:
        loss = nn.CrossEntropyLoss()

    if clamper is None:
        clamper = ut.imgClamper


    x = inp.unsqueeze(0).clone()
    x.requires_grad = True

    y = net(x).argmax(dim=1)

    epsilon = epsilon / num_iter

    for _ in range(num_iter):

        zero_gradients(x)
        out = net.forward(x)

        _loss = loss(out, y)
        _loss.backward()

        update = epsilon * torch.sign(x.grad.data)

        x.data = x.data + update
        x.data = clamper(x.data)

    delta = (x.data - inp).resize_as_(inp)

    ret_payload = IGSM_Return_Payload(delta)
    adv_ex = x.data.resize_as_(inp)

    return adv_ex.detach(), ret_payload


def fgsm(net, inp, epsilon):
    """
    Computes the the FGSM attack.

    net (torch.nn.Module): the model to calculate an
        adversarial example for

    inp (torch.Tensor): the input to perturb
        NOTE: the dimnesion of this input should
        be one less than what the network expects
        (e.g. if the network takes minibatches of size
        10 x 3 x 32 x 32 the input the input should
        just be of dimension 3 x 32 x 32)

    loss (torch.nn.Module): the loss function
        to apply defaults to cross entropy loss,

    epsilon (float): the maximal perturbation to be applied
        to any one input feature

    Returns:
        See the return value of IGSM

    """

    return untargeted_igsm(net, inp, epsilon, num_iter=1)



class IGSM(base_attack.BaseAttack):
    """
    Class implementing the iterative Fast Gradient Sign Attack

    Args:
        net (torch.nn.Module): the model to calculate an
            adversarial example for

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        clamper (callable): a callable to apply clamping
            to perturbed input features should take just a tensor
            and return the clamped tensor. Defaults to appropriate
            clamping behavior for torchvision images i.e. in the
            range of [0,1]. If no transform is desired the utility
			torch module, IdentityTransform, in the utils module can
			be passed.

        **atack_args: keyworded arguments which will be passed
            to the function, untargeted_igsm, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 normalizer=None,
                 clamper=None,
                 **attack_args):

        if clamper is None:
            clamper = ut.imgClamper

        self._params = ut.ParamDict(**attack_args)
        self.params['clamper'] = clamper

        self._attack_func = untargeted_igsm

        if normalizer is not None:
            self._model = nn.Sequential(normalizer, net)
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


class FGSM(IGSM):
    """
    Class implementing the FGSM attack

    Args:
        net (torch.nn.Module): the model to calculate an
            adversarial example for

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        clamper (callable): a callable to apply clamping
            to perturbed input features should take just a tensor
            and return the clamped tensor. Defaults to appropriate
            clamping behavior for torchvision images i.e. in the
            range of [0,1]. If no transform is desired the utility
			torch module, IdentityTransform, in the utils module can
			be passed.

        **atack_args: keyworded arguments which will be passed
            to the function, untargeted_igsm, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 normalizer=None,
                 clamper=None,
                 **attack_args):

        super(FGSM, self).__init__(net,
                                   normalizer=normalizer,
                                   clamper=clamper,
                                   **attack_args)

        self.params['num_iter'] = 1
        self.params.freeze_attr('num_iter')
