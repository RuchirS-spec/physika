import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
import sympy as sp

# === Program ===
x = sp.Symbol('x')
y = sp.Symbol('y')
u = sp.Function('u')
print(x)
f = ((x ** 2) + (y ** 2))
print(f)
expr = u(x, y)
print(expr)
print(f.subs([(x, 3.0), (y, 4.0)]))
f = (((x ** 3) + ((2 * x) ** 2)) + x)
print(sp.diff(f, x))
expr = ((x ** 2) + (y ** 2))
f = (lambda *args: sp.lambdify([x, y], expr, modules='torch')(*[torch.as_tensor(a).float() if not isinstance(a, torch.Tensor) else a for a in args]))
print(f(3.0, 4.0))
eq = sp.Eq(((2.0 * x) + 3.0), 7.0)
print(sp.solve(eq, x))