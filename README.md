# powell_optimization
realization of powell's method of conjugated directions.
included several methods of linear optimizations for this method.
```python
#  example
from base import Problem
from linear_optimization import ParabolicInterpolation
from powell import PowellMethod
class Func(Problem):
  def __init__(self):
    super().__init__(size=2)
  
  def f(p):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
func = Func()
pi = ParabolicInterpolation()
pwl = PowellMethod(pi)
print(pwl.minimize(func))
```
