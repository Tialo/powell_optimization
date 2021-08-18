from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from base import LineSearchMethod


class SplineSearch(LineSearchMethod):
    """
    Method of finding minimum of function by interpolating
    it with cubic splines, which built at randomly chosen points.
    """
    def __init__(self, point_number: int = None, splines: int = None, attempts: int = None):
        """
        :param point_number: number of randomly chosen points
        :param splines: number of addition splines
        :param attempts: number of independent tries of building a splines
        """
        if point_number is None:
            point_number = 50
        self.point_number = point_number
        if splines is None:
            splines = 5
        self.splines = splines
        if attempts is None:
            attempts = 2
        self.attempts = attempts

    def search(self, problem, x0, f0, d):
        """
        finding minimum of spline rebuilt {splines} times
        """
        non_zero_index = abs(d) > 1e-16
        ldv = (problem.l[non_zero_index] - x0[non_zero_index]) / d[non_zero_index]
        rdv = (problem.r[non_zero_index] - x0[non_zero_index]) / d[non_zero_index]
        negative_index = d[non_zero_index] < 0
        ldv[negative_index], rdv[negative_index] = rdv[negative_index], ldv[negative_index]
        l = np.max(ldv)
        r = np.min(rdv)

        f = problem.f
        min_value = np.inf
        p_min = l

        if r - l < 1e-16:
            p_min = (r + l) / 2
            return p_min, f(x0 + p_min * d)

        for i in range(self.attempts):
            points = [l, *np.random.uniform(l, r, self.point_number), r]
            points = np.array([(point, f(x0 + point * d)) for point in points])
            if not np.array([abs(points[:, 0]) < 1e-6]).any():
                points = np.append(points, np.array([[0.0, f0]]), axis=0)
            inds = np.argsort(points[:, 0])
            points = points[inds]
            rv = np.array(points)
            points = rv[:, 0]
            values = rv[:, 1]
            cs = CubicSpline(points, values)
            der = cs.derivative()
            roots = der.roots()
            roots = roots[(roots >= l) & (roots <= r)]
            count = 0
            while self.splines > count:
                count += 1
                for root in roots:
                    if not np.array([abs(rv[:, 0] - root) < 1e-6]).any():
                        value = f(x0 + root * d)
                        rv = np.insert(rv, rv[:, 0].searchsorted(root), (root, value), axis=0)
                points = rv[:, 0]
                values = rv[:, 1]
                cs = CubicSpline(points, values)
                der = cs.derivative()
                roots = der.roots()
                roots = roots[(roots >= l) & (roots <= r)]
            ind_min = np.argmin(rv[:, 1])
            value = rv[ind_min]
            if value[1] < min_value:
                min_value = value[1]
                p_min = value[0]
        return p_min, min_value


class ParabolicInterpolation(LineSearchMethod):
    """
    Method of finding minimum of function by interpolating it with quadratic parabolas.
    """
    def __init__(self, grow_limit: float = None, eps: float = None, initial_step: float = None):
        if grow_limit is None:
            grow_limit = 100.0
        self.grow_limit = grow_limit
        if eps is None:
            eps = 1e-6
        self.eps = eps
        if initial_step is None:
            initial_step = 1e-6
        self.initial_step = initial_step

    def initialize_brackets(self, f, x0, d, f0):
        gold = (np.sqrt(5) + 1) / 2
        t = 1e-20
        ax, bx = 0.0, self.initial_step
        fa, fb = f0, f(x0 + bx * d)
        if fb > fa:
            ax, bx = bx, ax
            fa, fb = fb, fa
        cx = bx + gold * (bx - ax)
        fc = f(x0 + cx * d)
        while fb > fc:
            r = (bx - ax) * (fb - fc)
            q = (bx - cx) * (fb - fa)
            if abs(q - r) < t:
                s = np.sign(q - r) * t
            else:
                s = q - r
            u = bx - ((bx - cx) * q - (bx - ax) * r) / (2 * s)
            ulim = bx + self.grow_limit * (cx - bx)
            if (bx - u) * (u - cx) > 0.0:
                fu = f(x0 + u * d)
                if fu < fc:
                    ax = bx
                    bx = u
                    fa = fb
                    fb = fu
                    break
                elif fu > fb:
                    cx = u
                    fc = fu
                    break
                u = cx + gold * (cx - bx)
                fu = f(x0 + u * d)
            elif (cx - u) * (u - ulim) > 0.0:
                fu = f(x0 + u * d)
                if fu < fc:
                    bx, cx, u = cx, u, u + gold * (u - cx)
                    fb, fc, fu = fc, fu, f(x0 + u * d)
            elif (u - ulim) * (ulim - cx) >= 0.0:
                u = ulim
                fu = f(x0 + u * d)
            else:
                u = cx + gold * (cx - bx)
                fu = f(x0 + u * d)
            ax, bx, cx = bx, cx, u
            fa, fb, fc = fb, fc, fu
        return sorted([(fa, ax), (fb, bx), (fc, cx)], key=lambda x: x[1])

    def search(self, problem, x0, f0, d):
        f = problem.f
        u = np.inf
        fu = np.inf
        (fa, a), (fb, b), (fc, c) = self.initialize_brackets(f, x0, d, f0)
        num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
        den = 2 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))
        while abs(u - (b - num / den)) > self.eps:
            u = b - num / den
            fu = f(x0 + u * d)
            points = sorted([(fa, a), (fb, b), (fc, c), (fu, u)])[:-1]
            (fa, a), (fb, b), (fc, c) = sorted(points, key=lambda x: x[1])
            num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
            den = 2 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))
        return sorted([(fa, a), (fb, b), (fc, c), (fu, u)])[0][::-1]


if __name__ == '__main__':
    pass
