import abc
import torch
from itertools import repeat
from enum import Enum


ABC = abc.ABCMeta('ABC', (object,), {})


class BaseAttack(ABC):
    """
    Abstract base class for model based attacks.

    All attacks which attack a specific model should subclass this abstract
    base class.

    When sublcassing this ABC three properties (i.e. using the
    @property decorator) are required to be defined: attack_type, model,
    params, and attack_func.

    'attack_type' should be one of the enum values of the class AttackType

    'attack_func' the actual function implementing the attack in the case
    of either of the targeted attacks there should be three positional
    arguments: the model, the input to perturb, and the index of the targeted
    class all remaining arguments should be named arguments denoting parameter
    specific attacks. In the case of an untargeted attack the only difference
    is that there should only be two positional arguments, the model, and
    the input to perturb.

    'model' should return the torch nn.Module that is being attacked

    'params' in all current cases is an instance of ParamDict (defined in
    utils.py) but any mutable mapping should work and this should contain
    named parameters that will be passed to the function returned by 'attack_func'
    """

    def batch_attack(self, inps, target_idxs=None, **attack_args):
        """
        Function to generate adversarial attacks for a batch of inputs.

        Args:
            inps (torch.Tensor): the batch of inputs to attack

            target_idxs (torch.Tensor, optional): a 1d tensor of the class labels
                the total amount should match the first dim of inps, if not
                given a value for a targeted increase style attack then the
                class that the network ranks as #2 will be guessed, if not
                given a value for target_idxs it will guess the top class

            **attack_args: named arguments that will be passed to each
                invocation of the attack function for the given class
        """
        if target_idxs is None and self.attack_type == AttackType.targeted_increase:
            target_idxs = targeted_guess(self.model, inps)

        elif not target_idxs and self.attack_type == AttackType.targeted_decrease:
            target_idxs = untargeted_guess(self.model, inps)

        if self.requires_target:
            assert inps.shape[0] == target_idxs.shape[0]

            results = (self.single_attack(inp, target_idx, **attack_args)
                       for inp, target_idx in zip(inps, target_idxs))
        else:
            results = (self.single_attack(inp, **attack_args) for inp in inps)

        advs, payloads = zip(*results)
        payloads = list(payloads)
        advs = list(advs)

        return torch.stack(advs), payloads


    def single_attack(self, inp, target_idx=None, **attack_args):
        """
        Function to generate adversarial attacks for a single input

        Args:
            inps (torch.Tensor): the input to attack, should have one less
                dimension then the model expects (e.g. if the model takes
                CIFAR-10 inputs of size ?x3x32x32 inp should be of size
                3x32x32)

            target_idx (torch.Tensor): a 1d tensor single element
                tensor of the class label to target, must be specified if the
                attack is of type targeted_increase or targeted_decrease

            **attack_args: named arguments that will be passed to the
                invocation of the attack function for the given class
        """
        if target_idx is None and self.requires_target:
            raise ValueError(("If the attack requires a target label it must"
                              " be included as the second argument"))

        self.params.update(**attack_args)

        if self.requires_target:
            return self.attack_func(self.model, inp, target_idx, **self.params)

        else:
            return self.attack_func(self.model, inp, **self.params)

    @property
    def requires_target(self):
        """
        Whether the attack requires a target index or not
        """
        return self.attack_type == AttackType.targeted_increase or \
               self.attack_type == AttackType.targeted_decrease

    @abc.abstractproperty
    def attack_type(self):
        """
        The type of attack, specified by the values of the enum, AttackType
        """
        pass

    @abc.abstractproperty
    def params(self):
        """
        A mutable mapping of named arguments to be passed to attack_func
        """
        pass

    @abc.abstractproperty
    def attack_func(self):
        """
        The actual function implementing the attack.
        """
        pass

    @abc.abstractproperty
    def model(self):
        """
        The model being attacked.
        """
        pass


class AttackType(Enum):
    """
    Enum denoting the type of attack.

    Untargeted is meant for attacks that truly do not need
    any sort of target class (e.g. DeepFool)

    Targeted increase is what is referred to as a 'targeted'
    attack in most the literature and refers to attacks that
    specifically increase the score of a target class.

    Targeted decrease refers to attacks that are sometimes
    referred to as 'untargeted' in the literature but which
    actually can take and class label to decrease the score
    of.
    """
    untargeted = 1
    targeted_decrease = 2
    targeted_increase = 3


def targeted_guess(net, inps):
    """
    Returns the class with the second highest score
    used as a helper for batch attack.
    """
    outs = net(inps)
    assert len(outs.shape) == 2

    return outs.topk(k=2)[1][:, 1]


def untargeted_guess(net, inps):
    """
    Returns the top predicted class used as
    a helper in batch attack.
    """
    outs = net(inps)
    assert len(outs.shape) == 2

    return outs.argmax(dim=1)


