from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.autograd.gradcheck import zero_gradients
from . import utils as ut
from collections import namedtuple
from . import base_attack

class TanhTransform(nn.Module):
    """
    Computes the tanh transform used to
    remove box constraints from C&W paper

    NOTE: This reparamterization trick is
    highly numerically unstable even for small-ish
    values so should really only be used
    for inputs that are bounded above or below
    by relatively small values

    Args:
        xmin (float or torch.Tensor):
            the lower bound for input values
            should either be a float or broadcastable
            with the input tensor where each element
            in the tensor corresponds to the lower
            bound of an input feature

        xmax (float or torch.Tensor):
            the lower bound for input values
            should either be a float or broadcastable
            with the input tensor where each element
            in the tensor corresponds to the upper
            bound of an input feature

    """
    def __init__(self, xmin=0, xmax=1):
        super(TanhTransform, self).__init__()

        delta = xmax - xmin

        self.delta_2 = delta / 2

        self.xmax = xmax
        self.xmin = xmin

    def forward(self, x):

        out = (x.tanh() + 1) * self.delta_2 + self.xmin
        return out

    def invert_forward(self, x):
        z = (x - self.xmin) / self.delta_2 - 1
        return arctanh(z)


# named tuples to hold the extra return values from the attacks
CWL2_Return_Payload = namedtuple("CW_Return_Payload", ["delta", "w"])
CWLInfty_Return_Payload = namedtuple("CW_Return_Payload",
                                     ["delta", "w", "num_passes", "c", "success"])

def targeted_cwl2(net, inp, target,
                  box_transform=None,
                  num_iter=5,
                  c=0.5,
                  kappa=0,
                  optim_payload=None):
    r"""
    Computes an adversarial example using the
    L_2 Carlini and Wagner attack from the paper:
    "Towards Evaluating the Robustness of Neural Networks"

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        inp (torch.Tensor): the input to perturb
            NOTE: the dimnesion of this input should
            be one less than what the network expects
            (e.g. if the network takes minibatches of size
            10 x 3 x 32 x 32 the input the input should
            just be of dimension 3 x 32 x 32)

        target (int): the target class to transform x into

        box_transform (callable): a callable to transform
            input features as described in the original
            paper, in their formulation this functionality
            is provided by the Tanh transform specified on
            page 7 of the paper. If there is no need for such
            a transform the pytorch nn.Module called
            IdentityTransform can be passed for this parameter
            Defaults to TanhTransform() defined in this file,
            which assumes every input feature is in [0,1]
            (as torchvision images are)

        c (float): the weight attached to the loss
            function, f,

        num_iter (int): the number of steps to run
                    the optimizer for

        kappa (float): see page 9 of the paper

        optim_payload (mapping): the keyword arguments
            to pass to the ADAM optimizer, defaults to just
            setting the lr to 0.01, for more arguments
            seed the pytorch documentation for ADAM

    Returns:

        torch.Tensor: The computed adversarial example
            of the same dimension as input (inp)

        namedtuple: Used to return additional information

            contains:

            delta: diff between the adversarial example
                and the original input

            w: adversarial image under the tanh transform
    """
    if box_transform is None:
        box_transform = TanhTransform()

    if optim_payload is None:
        optim_payload = {'lr': 0.01}

    w = box_transform.invert_forward(inp).unsqueeze(0)
    w.requires_grad = True

    loss = CWL2_Loss()
    optimizer = optim.Adam([w], **optim_payload)

    for _ in range(num_iter):
        zero_gradients(net)
        zero_gradients(loss)
        logits = net(w)

        _loss = loss(w, target, inp, logits, box_transform, c=c, kappa=kappa)

        _loss.backward()
        optimizer.step()

    adv_ex = box_transform(w).detach().resize_as_(inp)
    delta = (adv_ex - inp).detach().resize_as_(inp)

    ret_payload = CWL2_Return_Payload(delta, w.detach())

    return adv_ex, ret_payload


class Carlini_Wagner_L2(base_attack.BaseAttack):
    """
    Class implementing the Carlini and Wagner L2 norm attack

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        box_transform (callable): a callable to transform
            input features as described in the original
            paper, in their formulation this functionality
            is provided by the Tanh transform specified on
            page 7 of the paper. If there is no need for such
            a transform the pytorch nn.Module called
            IdentityTransform can be passed for this parameter
            Defaults to TanhTransform() defined in this file,
            which assumes every input feature is in [0,1]
            (as torchvision images are)

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        **atack_args: keyworded arguments which will be passed
            to the function, targeted_cwl2, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 box_transform=None,
                 normalizer=None,
                 **attack_args):

        if box_transform is None:
            box_transform = TanhTransform()

        model_trans = copy.deepcopy(box_transform)
        self._params = ut.ParamDict(**attack_args)
        self._params.box_transform = copy.deepcopy(model_trans)

        self._attack_func = targeted_cwl2

        if normalizer is not None:
            self._model = nn.Sequential(model_trans, normalizer, net)
        else:
            self._model = nn.Sequential(model_trans, net)

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


def targeted_cwlinfty(net, inp, target,
                      box_transform=None,
                      num_iter=5,
                      c=1e-2,
                      kappa=0,
                      c_thresh=10e4,
                      optim_payload=None):
    r"""
    Computes an adversarial example using the
    L_\infty Carlini and Wagner attack from the paper:
    "Towards Evaluating the Robustness of Neural Networks"

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        inp (torch.Tensor): the input to perturb
            NOTE: the dimnesion of this input should
            be one less than what the network expects
            (e.g. if the network takes minibatches of size
            10 x 3 x 32 x 32 the input the input should
            just be of dimension 3 x 32 x 32)

        target (int): the target class to transform x into

        box_transform (callable): a callable to transform
            input features as described in the original
            paper, in their formulation this functionality
            is provided by the Tanh transform specified on
            page 7 of the paper. If there is no need for such
            a transform the pytorch nn.Module called
            IdentityTransform can be passed for this parameter
            Defaults to TanhTransform() defined in this file,
            which assumes every input feature is in [0,1]
            (as torchvision images are)

        c (float): the weight attached to the loss
            function, f, the value starts at the one
            given and doubles if a successful adversarial
            example can't be found

        c_thresh (float): the maximum threshold
            for c. Once the value of c exceeds c_thresh
            the search stops

        num_iter (int): the number of steps to run
            the optimizer for

        kappa (float): see page 9 of the paper

        optim_payload (mapping): the keyword arguments
            to pass to the ADAM optimizer, defaults to just
            setting the lr to 0.01, for more arguments
            seed the pytorch documentation for ADAM

    Returns:

        torch.Tensor: The computed adversarial example
            of the same dimension as input (inp)

        namedtuple: Used to return additional information

            contains:

            delta: diff between the adversarial example
                and the original input

            w: adversarial image under the tanh transform
    """
    if box_transform is None:
        box_transform = TanhTransform()

    if optim_payload is None:
        optim_payload = {'lr': 0.01}

    w = box_transform.invert_forward(inp).unsqueeze(0)
    w.requires_grad = True

    tau = 1
    soft_norm = 1 # really any non-zero value would work here

    # factors by which to increase tau/c each iter
    tau_factor = 0.9
    c_factor = 2

    loss = CWLInfty_Loss()
    optimizer = optim.Adam([w], **optim_payload)

    predicted = ut.predict(net, w)

    num_passes = 0

    while soft_norm != 0 and c < c_thresh and predicted != target:

        # NOTE: this is implementation uses the previous value of
        # the pertrubed input to warmstate the process

        for iter in range(num_iter):
            zero_gradients(net)
            zero_gradients(loss)
            logits = net(w)

            soft_norm = soft_maxnorm(box_transform(w) - inp, tau)

            _loss = loss(w, target, logits, soft_norm,
                         box_transform, c=c, kappa=kappa)

            _loss.backward()
            optimizer.step()

        tau *= tau_factor
        c *= c_factor
        predicted = ut.predict(net, w)

        num_passes += 1

    adv_ex = box_transform(w).detach().resize_as_(inp)
    delta = (adv_ex - inp).detach().resize_as_(inp)

    ret_payload = CWLInfty_Return_Payload(delta, w,
                                          num_passes, c,
                                          int(predicted == target))

    return adv_ex, ret_payload


class Carlini_Wagner_LInfinity(base_attack.BaseAttack):
    """
    Class implementing the Carlini and Wagner L_infty norm attack

    Args:
        net (nn.Module): the model to calculate an
            adversarial example for

        box_transform (callable): a callable to transform
            input features as described in the original
            paper, in their formulation this functionality
            is provided by the Tanh transform specified on
            page 7 of the paper. If there is no need for such
            a transform the pytorch nn.Module called
            IdentityTransform can be passed for this parameter
            Defaults to TanhTransform() defined in this file,
            which assumes every input feature is in [0,1]
            (as torchvision images are)

        normalizer (torch.nn.Module): a class applying input
            normalization to the network, net, can be user
            created or if the inputs will be images the
            utility torch.nn module, Normalize, defined in
            attacks/utils can be used which applies per channel
            normalization

        **atack_args: keyworded arguments which will be passed
            to the function, targeted_cwlinfty, see its docstring
            for details on available attack parameters
    """
    def __init__(self, net,
                 box_transform=None,
                 normalizer=None,
                 **attack_args):

        if box_transform is None:
            box_transform = TanhTransform()

        model_trans = copy.deepcopy(box_transform)
        self._params = ut.ParamDict(**attack_args)
        self._params.box_transform = copy.deepcopy(model_trans)

        self._attack_func = targeted_cwlinfty

        if normalizer is not None:
            self._model = nn.Sequential(model_trans, normalizer, net)
        else:
            self._model = nn.Sequential(model_trans, net)

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


class CWL2_Loss(nn.Module):
    """
    A class implementing the loss for the C&W L2 attack
    """

    def __init__(self):
        super(CWL2_Loss, self).__init__()

    def forward(self, w, y, x_orig, logits, box_transform, c=0.5, kappa=0):
        """
        Forward pass for the C&W L_2 loss,
        see page 9 of the paper for reference.

        Args:
            w (torch.Tensor): The input after
                the tanh reparameterization trick from
                the paper has been applied

            y (torch.Tensor or int): the target index

            x_orig (torch.Tensor): the unmodified image

            logits (torch.Tensor): the output logits of
                the network

            c (float): the weight attached to the loss
                function

            kappa (float): see page 9 of the paper
        """
        x = box_transform(w)

        norm = (x - x_orig).norm(p=2) ** 2
        f = cwl2_lossf(logits, y, kappa)

        return norm + c * f

class CWLInfty_Loss(nn.Module):
    """
    A class implementing the loss for the
    C&W L-Infinity attack
    """

    def __init__(self):
        super(CWLInfty_Loss, self).__init__()

    def forward(self, w, y, logits, soft_norm, box_transform, c=0.5, kappa=0):
        """
        Forward pass for the C&W L_Infinity loss,

        Args:
            w (torch.Tensor): The input after
                the tanh reparameterization trick from
                the paper has been applied

            logits (torch.Tensor): the output logits of
                the network

            y (torch.Tensor or int): the target index

            soft_norm (float): the value of the soft
                infinity norm as described in page 10
                of the paper

            c (float): the weight attached to the loss
                function

            kappa (float): see page 9 of the paper
        """
        x = box_transform(w)

        f = cwl2_lossf(logits, y, kappa)

        return soft_norm + c * f


def arctanh(x, eps=1e-6):
    """
    Calculates the inverse hyperbolic tangent.
    """
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def soft_maxnorm(delta, tau):
    return torch.clamp(delta - tau, min=0).sum()


def cwl2_lossf(logits, target, kappa=0):
    """
    Computes the non-norm portion of the C&W L_2 attack

    This is a direct translation of the formula on
    page 9 of the paper "Towards Evaluating the
    Robustness of Neural Networks"

    Args:
        logits (torch.Tensor): 2d torch tensor of output
            logit units of a model (no softmax should be
            applied to the model)

        target (float): the target class whose score
            should be increased

        kappa (float): the parameter kappa from the
            paper (see pg. 9 for details.)
    """
    mask = torch.ones_like(logits, dtype=torch.uint8)

    target_val = logits[:, target]
    non_targets = logits[mask]
    max_non_target = non_targets.max()

    return torch.clamp(max_non_target - target_val, min=-kappa)
