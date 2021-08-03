from problems import *
from linear_search import *
import numpy as np


class PowellMethod:

    def __init__(self, ls: LineSearchMethod, restarts: int = None,
                 eps: float = None, iterations: int = None, always_change_basis: bool = False):
        self.ls = ls
        if restarts is None:
            restarts = 1
        self.restarts = restarts
        if eps is None:
            eps = 1e-6
        self.eps = eps
        if iterations is None:
            iterations = 100
        self.iterations = iterations
        self.always_change_basis = always_change_basis

    def minimize(self, problem: Problem, x0: np.ndarray) -> (np.ndarray, float):
        f = problem.f
        ls = self.ls
        if x0 is None:
            x0 = np.zeros(problem.size)
        elif x0.dtype != np.float64:
            x0 = x0.astype(np.float64)
        it = 0
        k = problem.size
        xi = np.copy(x0)
        fi = f(xi)
        while it < self.restarts:
            d = np.array([np.array([1.0 if i == j else 0.0 for i in range(k)]) for j in range(k)])
            xi = np.copy(x0)
            fi = f(xi)
            iteration = 0
            while iteration < self.iterations:
                x0i = np.copy(xi)
                f0i = fi
                delta_k = 0
                ind = 0
                for i, di in enumerate(d):
                    diff = fi
                    alpha, fi = ls.search(problem, xi, di)
                    diff -= fi
                    if diff > delta_k:
                        delta_k = diff
                        ind = i
                    xi += alpha * di
                dn = xi - x0i
                if np.all(np.abs(d) < self.eps) or np.linalg.norm(dn) < 0.1 * self.eps:
                    break
                if not self.always_change_basis:
                    f1, f2, f3 = f0i, fi, f(2 * xi - x0i)
                    if f3 >= f1 or (f1 - 2 * f2 + f3) * (f1 - f2 - delta_k) ** 2 >= 0.5 * delta_k * (f1 - f3) ** 2:
                        if f3 < f2:
                            xi = 2 * xi - x0i
                            fi = f3
                        iteration += 1
                        continue
                alpha, fi = ls.search(problem, xi, dn)
                xi += alpha * dn
                d = np.append(np.delete(d, ind, axis=0), np.array([dn]), axis=0)
                if np.all(np.abs(d) < self.eps) or np.linalg.norm(xi - x0i) < 0.1 * self.eps:
                    break
                iteration += 1
            x0 = np.copy(xi)
            it += 1
        return fi, xi


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
    rs = RandomSearch(60)
    res = RandomSearch(150)

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
