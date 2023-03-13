
import torch
import matplotlib.pylab as plt


a = torch.Tensor([0.7023, 0.6493, 0.6815])
a = a.expand((500,500,3))

plt.imsave('asd.png',a.numpy())