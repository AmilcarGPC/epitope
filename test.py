import torch
import torch_sparse
import torch_scatter

print("Torch version:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
print(torch.version.cuda)
print("torch_sparse version:", torch_sparse.__version__)
print("torch_scatter version:", torch_scatter.__version__)