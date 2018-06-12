from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from . import utils as ut
import operator as op
from collections import namedtuple
from . import base_attack


JSMA_Return_Payload = namedtuple("JSMA_Return_Payload", ["delta", "num_passes"])


def untargeted_jsma(net, inp, target_idx, **attack_args):
    """
    Performs the un-targeted generic JSMA attack

    See function jsma in this file for an explanation of the
    arguments and return value.
    """

    return jsma(net, inp, untargeted_saliency_map, target_idx, op.eq,
                **attack_args)


def targeted_jsma(net, inp, target_idx, **attack_args):
    """
    Performs the targeted generic JSMA attack

    See function jsma in this file for an explanation of the
    arguments and return value.
    """

    return jsma(net, inp, targeted_saliency_map, target_idx, op.ne,
                **attack_args)


class JSMA_V(base_attack.BaseAttack):
    """
    Abstract base class for the vanilla JSMA attack
    """
    def __init__(self, net,
                 normalizer=None,
                 **attack_args):

        self._params = ut.ParamDict(**attack_args)

        if normalizer is not None:
            self._model = nn.Sequential(normalizer, net)
        else:
            self._model = net

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params


class JSMA_VD(JSMA_V):
    """
    Class implementing the vanilla untargeted JSMA attack.

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        **attack_args: keyworded arguments which will be passed
            to the function, deepfool, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 normalizer=None,
                 **attack_args):

        super(JSMA_VD, self).__init__(net, normalizer=normalizer, **attack_args)
        self._attack_func = untargeted_jsma

    @property
    def attack_func(self):
        return self._attack_func

    @property
    def attack_type(self):
        return base_attack.AttackType.targeted_decrease


class JSMA_VI(JSMA_V):
    """
    Class implementing the targeted vanilla JSMA attack.

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
            to the function, targeted_jsma, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 normalizer=None,
                 **attack_args):

        super(JSMA_VI, self).__init__(net, normalizer=normalizer, **attack_args)
        self._attack_func = targeted_jsma

    @property
    def attack_func(self):
        return self._attack_func

    @property
    def attack_type(self):
        return base_attack.AttackType.targeted_increase


def jsma(net, inp, saliency_map, target_idx, comp_op,
         theta=10./255, gamma=100./255,
         norm=np.inf):
    """
    Performs a variant of the JSMA attack

    Args:
        net (nn.Module): the network perform the attack
            on
        inp (torch.Tensor): the input to calculate an
            adversarial attack for
        saliency_map (function): the function used to
            compute the saliency map
        target_idx (int): the index of the target
            class
        comp_op (function): a function that returns true
            while the nets output either matches the target
            (if its being decreased) or when they don't
            (in the case where the target classes' score
            is being increased

        theta (float): how much to increase an input feature by

        gamma (float): the maximum value allowed

        norm (int): an integer representing which p-norm
            to use when calculting the distortion between
            the adversarial example and the original input

    Returns:

        torch.Tensor: The computed adversarial example
            of the same dimension as input (inp)

        namedtuple: Used to return additional information

            contains:

            delta: diff between the adversarial example
                and the original input

            num_passes: the number of passes the algorithm
                ran for
    """
    inp_orig = inp.clone()
    inp = inp.clone().unsqueeze(0)
    inp.requires_grad = True

    linear_view = inp.view(inp.numel()).clone()
    delta = 0

    num_passes = 0

    while comp_op(ut.predict(net, inp), target_idx) and delta < gamma:
        out = net(inp)
        jacobian = compute_jacobian(inp, out)

        smap = saliency_map(jacobian, target_idx)

        val, idx = smap.max(dim=0)

        linear_view.data[idx] += theta

        delta_z = (linear_view - inp_orig.view(inp.numel())).detach()
        delta = np.linalg.norm(delta_z, ord=norm)

        inp.data = linear_view.data.view(inp.data.shape)

        num_passes += 1

    adv_ex = inp.detach().resize_as_(inp_orig)
    delta = adv_ex - inp_orig
    return_payload = JSMA_Return_Payload(delta, num_passes)

    return adv_ex, return_payload


def untargeted_saliency_map(jacobian, target_idx, search_space=1):
    r"""
    Computes a saliency map for targeted attacks.

    Args:
        jacobian (Tensor or ndarray): 2d jacobian matrix
            should be the result of calling `jacobian`
            the first dim should index into features and
            the second into output labels

        target_idx (Int): the target class whose score
            should be decreased

        search_space (Tensor or ndarray): 1d mask array
            each index in the search space array should
            correspond to a index in the linearlized
            input space, the indices corresponding to
            input features that should be excluded
            from the saliency map location should
            be set to 0


    Returns:
        Tensor: an array of values where index i corresponds
            to the saliency of input feature i
    """

    # pre-conditions
    # make sure jacobian is of two dimensions
    # make sure target_idx is within bounds (target_idx <= num_columns)
    assert len(jacobian.shape) == 2
    assert target_idx <= jacobian.shape[1]

    target_scores = jacobian[:, target_idx]
    other_scores = jacobian.sum(dim=1) - target_scores

    ts_idx = (target_scores < 0).type(target_scores.type())

    os_idx = (other_scores > 0).type(other_scores.type())

    mask = os_idx * ts_idx * search_space

    return target_scores.abs() * other_scores * mask


def targeted_saliency_map(jacobian, target_idx, search_space=1):
    r"""
    Computes a saliency map for targeted attacks.

    Args:
        jacobian (Tensor or ndarray): 2d jacobian matrix
            should be the result of calling `jacobian`
            the first dim should index into features and
            the second into output labels

        target_idx (Int): the target whose score should
            be increased

        search_space (Tensor or ndarray): 1d mask array
            each index in the search space array should
            correspond to a index in the linearlized
            input space, the indices corresponding to
            input features that should be excluded
            from the saliency map location should
            be set to 0

    Returns:
        Tensor: an array of values where index i corresponds
            to the saliency of input feature i
    """

    # pre-conditions
    # make sure jacobian is of two dimensions
    # make sure target_idx is within bounds (target_idx <= num_columns)
    assert len(jacobian.shape) == 2
    assert target_idx <= jacobian.shape[1]

    target_scores = jacobian[:, target_idx]
    other_scores = jacobian.sum(dim=1) - target_scores

    ts_idx = (target_scores > 0).type(target_scores.type())
    os_idx = (other_scores < 0).type(other_scores.type())

    mask = os_idx * ts_idx * search_space

    return target_scores * other_scores.abs() * mask


def format_jacobian(jacobian):
    """
    Reformats the calculated Jacobian to a 2D array

    This function squashes the intermediate jacobian
    created by the func jacobian into a format that
    notationall matches what's in the paper.
    """

    output_dim = jacobian.shape[0]
    return jacobian.view(output_dim, -1).t()


def compute_jacobian(inputs, outputs, paper_format=True):
    """
    Computes the jacobian of an input value,

    Adapted from here: https://discuss.pytorch.org/t/2052

    Args:
        inputs: The input to calculate the jacobian of

        outputs: The output of a model w.r.t. inputs

        paper_format (optional): returns the jacobian
        in the format given in the paper where it is
        a flat 2d matrix and rows correspond to features
        and columns to outputs.

    Returns:
        A matrix where the leading dimension is the
        number of output features and the remaining
        dimensions match the inputs' dimension.

        Due to this jacobian[j] will be a matrix with
        dimensions equal to the input where each individual
        element will be the derivative of output j w.r.t that
        input element.
    """

    # need to be able to compute gradients w.r.t input
    assert inputs.requires_grad
    num_classes = outputs.size()[1]

    # output vector to hold the jacobian,
    jacobian = torch.zeros(num_classes, *inputs.size())

    # output should be of dimension 2
    grad_output = torch.zeros(*outputs.size())

    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        # zero_gradients(inputs)
        # inputs.zero_grad()

        # only calculate gradients w.r.t output i
        grad_output.zero_()
        grad_output[:, i] = 1

        # torch.autograd.grad(outputs, inputs)

        outputs.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

        inputs.grad.data.zero_()

    # NOTE: the result 'jacobian' matrix at this point is
    # actually not quite what's in the paper for that all
    # but the 1st dimension needs to be squashed
    # and the transpose needs to be taken

    return format_jacobian(jacobian) if paper_format else jacobian
