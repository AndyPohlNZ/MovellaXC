"""
Implements a pSpline via GCV
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intrp
from scipy.optimize import minimize
import copy


class PSpline:
    def __init__(self, x, y, bDeg, nK=None, pDeg=3, pad=0):
        self.bDeg = bDeg
        self.pDeg = pDeg
        if nK == None:
            self.nK = np.int64(len(x) / 5)
        else:
            self.nK = nK
        self.knots = np.linspace(np.min(x), np.max(x), self.nK)

        self.bspline = copy.deepcopy(self.__gen_Bspline(x, y, pad))

    def __penalty(self, K):
        """Construct a penalty matrix of pth order differences"""
        p = copy.deepcopy(self.pDeg)
        # K = copy.deepcopy(self.nK)
        D = np.diff(np.identity(K + p), p, axis=0)
        S = np.transpose(D) @ D
        return S[(p):, (p):]

    def __bbasis(self, x, knots):
        p = copy.deepcopy(self.bDeg)
        numpyknots = np.asarray(
            [*np.repeat(np.min(knots), p), *knots, *np.repeat(np.max(knots), p)]
        )
        basis = np.zeros((len(x), len(knots) + p - 1))  # check if p is deg or order
        for i in range(len(knots) + p - 1):
            basis[:, i] = intrp.BSpline(
                numpyknots,
                (np.arange(len(knots) + p - 1) == i).astype(np.float64),
                p,
                extrapolate=False,
            )(x)
        return basis

    def __gcv(self, lam, y, B, Pmat):
        lam = np.exp(lam)
        Bt = np.transpose(B)
        hat_mat = B @ np.linalg.inv(Bt @ B + lam * Pmat) @ Bt
        mu = B @ np.linalg.inv(Bt @ B + (lam * Pmat)) @ Bt @ y
        V = np.sqrt(np.mean((y - mu / (1 - np.diagonal(hat_mat))) ** 2))
        return V

    def __get_lam(self, x0, y, B, Pmat):
        res = minimize(
            self.__gcv,
            x0,
            method="nelder-mead",
            args=(y, B, Pmat),
            options={"xatol": 1e-8, "disp": False},
        )
        return res.x

    def __pen_likelihood(self, lam, y, B, Pmat):
        xbar = np.row_stack((B, np.sqrt(lam) * Pmat))
        xbart = np.transpose(xbar)
        ybar = np.asarray((*y, *np.repeat(0, Pmat.shape[0])))
        coef = np.linalg.inv(xbart @ xbar) @ xbart @ ybar
        return coef

    def __gen_Bspline(self, x, y, pad):
        bDeg = copy.deepcopy(self.bDeg)
        pDeg = copy.deepcopy(self.pDeg)
        if pad > 0:
            dt = np.mean(np.diff(x))
            x = np.asarray(
                [
                    *np.linspace(x[0] - pad * dt, x[0], pad),
                    *x,
                    *np.linspace(x[-1], x[-1] + pad * dt, pad),
                ]
            )
            dy0 = y[1] - y[0]
            dy1 = y[-1] - y[-2]
            # y = np.asarray(
            #     [
            #         *np.linspace(y[0] - pad * dy0, y[0], pad),
            #         *y,
            #         *np.linspace(y[-1], y[-1] + pad * dy1, pad),
            #     ]
            # )
            y = np.asarray(
                [
                    *np.repeat(y[0], pad),
                    *y,
                    *np.repeat(y[-1], pad),
                ]
            )

        nK = copy.deepcopy(self.nK)
        breakpoints = np.linspace(np.min(x), np.max(x), nK)
        basis = self.__bbasis(x, breakpoints)
        pMat = self.__penalty(len(breakpoints) + self.bDeg - 1)
        lam = self.__get_lam(np.log(100), y, basis, pMat)
        coef = self.__pen_likelihood(np.exp(lam), y, basis, pMat)
        numpyknots = np.asarray(
            [
                *np.repeat(np.min(breakpoints), bDeg),
                *breakpoints,
                *np.repeat(np.max(breakpoints), bDeg),
            ]
        )

        return intrp.BSpline(numpyknots, coef, bDeg)


if __name__ == "__main__":

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.01, len(x))

    bsp = PSpline(x, y, 5, pDeg=3, pad=10)

    ax = plt.subplot()
    ax.scatter(x, y)
    ax.plot(x, np.sin(x))
    ax.plot(x, bsp.bspline(x), color="red")
    ax.plot(x, bsp.bspline.derivative(1)(x), color="green")
    ax.plot(x, bsp.bspline.derivative(2)(x), color="red")

    plt.show()
