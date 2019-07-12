from network import vgg
import torch

models = getattr(vgg, 'vgg16')
model = models(False)


x = torch.randn((3,3,224,224))
logit_A, logit_B = model(x)

print(logit_A.shape, logit_B.shape)

# a = torch.range(0,49)
# a = a.view(5,-1)
#
# print(a)
# b = a[range(5),[0,1,4,5,6]]
# print(b)
# print(b.size())
