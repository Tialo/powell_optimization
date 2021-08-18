from linear_optimization import *
from powell import PowellMethod
from problems import *
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    x1 = np.array([-2, 1])
    x2 = np.array([-13, 24])
    x3 = np.array([1, 0])

    ea = EasomProblem()
    c = CrossProblem()
    s = SchafferProblem()
    ht = HolderTableProblem()
    eh = EggHolderProblem()
    r = RosenbrockProblem()
    mc = McCormicProblem()
    b = BukinProblem()

    pi = ParabolicInterpolation()
    rs = SplineOptimization(60)
    res = SplineOptimization(150)

    par_pow = PowellMethod(pi)
    rand_pow = PowellMethod(rs)
    rand_pow2 = PowellMethod(res, always_change_basis=True, eps=1e-2)

    print(par_pow.minimize(r, x2))
    print(rand_pow.minimize(ht, x2))
    print(rand_pow.minimize(eh, x2))
    print(rand_pow.minimize(s, x2))
    print(rand_pow.minimize(c, x1))
    print(rand_pow2.minimize(ea, x2))
    print(rand_pow.minimize(mc, x3))
    print(par_pow.minimize(b, x1))
