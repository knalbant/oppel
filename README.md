# oppel

A toolbox to create adversarial attacks with PyTorch

## Install

From the base of this directory run python setup.py install

## Example Usage

From the demo directory

```python
import torch.nn
from torch.nn.functional import softmax
from torchvision import datasets

from oppel.attacks import migsm
import oppel.attacks.utils as ut

from cifar10model import cifar10, classes

net = cifar10()
norm = ut.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
net = nn.Sequential(norm, net)
net = net.eval()

attack = migsm.MIGSM_D(net, normalizer=norm)

test_data = datasets.CIFAR10(root='.', download=True, train=False, transform=tf.ToTensor())

X, y = random.choice(test_data)
x = X.unsqueeze(0)

adv_ex, return_payload = migsm.batch_attack(x)

prob, idx = softmax(net(X.unsqueeze(0)),1).max(dim=1)
print("Unperturbed Class: {}  with probability: {}".format(classes[idx], prob.item() * 100))

prob, idx = softmax(net(adv_ex.unsqueeze(0))),1).max(dim=1)
print("Perturbed Class: {}  with probability: {}".format(classes[idx], prob.item() * 100))
```

Features
--------
Both Python2.7 and Python3.4+ are supported.
