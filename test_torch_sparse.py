import torch
from torch_sparse import SparseTensor, diag

# Verifica que CUDA esté disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Crea algunos tensores simples para construir una SparseTensor
row = torch.tensor([0, 1, 2], device=device)
col = torch.tensor([1, 2, 0], device=device)
values = torch.tensor([1.0, 2.0, 3.0], device=device)
size = (3, 3)

# Crea la SparseTensor
st = SparseTensor(row=row, col=col, value=values, sparse_sizes=size)
print("SparseTensor creada:")
print(st)

# Prueba la función fill_diag para asegurarte de que no lanza error
filled = diag.fill_diag(st, torch.tensor(10.0, device=device))
print("SparseTensor con diagonal llenada:")
print(filled)