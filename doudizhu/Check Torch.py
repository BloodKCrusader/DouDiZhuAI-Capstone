import torch

print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())


x = torch.rand(5, 3)
print(x)