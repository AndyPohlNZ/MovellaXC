from numpy import asarray, min, zeros
from matplotlib.pyplot import get_cmap
from rpy2.robjects.packages import importr
from rpy2.situation import iter_info
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mvn
import numpy as np
import copy
import quaternion


# Default colors
CMAP = get_cmap("tab10")
# data directory
DATA_DIR = ""
# Subject table
SUBJECT_TABLE = DATA_DIR + "subjects.csv"

# Constants
GRAVITY = (
    9.80665  # m/s/s  # Acceleration due to gravity acts in the negative z direction...
)
MAX_EASTING = 900000  # m approximate maximum easting coordinate in m.

POLE_DENSITY = 0.192 / 1.5875  # kg/m of length
SKI_DENSITY = 0.706 / (2.01 * 0.04445)  # kg/m^2 of area (length x width)

SEGMENT_NAMES = [
    "l_foot",
    "l_lower_leg",
    "l_upper_leg",
    "r_foot",
    "r_lower_leg",
    "r_upper_leg",
    "l_hand",
    "l_forearm",
    "l_upper_arm",
    "r_hand",
    "r_forearm",
    "r_upper_arm",
    "lower_trunk",
    "mid_trunk",
    "upper_trunk",
    "head",
]

# segment proportional mass of total body mass
# See table 4.1 in DeLava (1995)
SEGMENT_MASS_MALE = {
    "head": 6.94 / 100,
    "upper_trunk": 15.96 / 100,
    "mid_trunk": 16.33 / 100,
    "lower_trunk": 11.17 / 100,
    "upper_arm": 2.71 / 100,
    "forearm": 1.62 / 100,
    "hand": 0.61 / 100,
    "upper_leg": 14.16 / 100,
    "lower_leg": 4.33 / 100,
    "foot": 1.37 / 100,
}
SEGMENT_MASS_FEMALE = {
    "head": 6.68 / 100,
    "upper_trunk": 15.45 / 100,
    "mid_trunk": 14.65 / 100,
    "lower_trunk": 12.47 / 100,
    "upper_arm": 2.55 / 100,
    "forearm": 1.38 / 100,
    "hand": 0.56 / 100,
    "upper_leg": 14.78 / 100,
    "lower_leg": 4.81 / 100,
    "foot": 1.29 / 100,
}
# Segment length proportions from full height
SEGMENT_LENGTH_MALE = {
    "head": 0.2033 / 1.741,
    "upper_trunk": 0.1707 / 1.741,
    "mid_trunk": 0.2155 / 1.741,
    "lower_trunk": 0.1457 / 1.741,
    "upper_arm": 0.2817 / 1.741,
    "forearm": 0.2689 / 1.741,
    "hand": 0.0862 / 1.741,
    "upper_leg": 0.4222 / 1.741,
    "lower_leg": 0.434 / 1.741,
    "foot": 0.2581 / 1.741,
}
SEGMENT_LENGTH_FEMALE = {
    "head": 0.2002 / 1.735,
    "upper_trunk": 0.1425 / 1.735,
    "mid_trunk": 0.2053 / 1.735,
    "lower_trunk": 0.1815 / 1.735,
    "upper_arm": 0.2751 / 1.735,
    "forearm": 0.2643 / 1.735,
    "hand": 0.0780 / 1.735,
    "upper_leg": 0.3685 / 1.735,
    "lower_leg": 0.4323 / 1.735,
    "foot": 0.2283 / 1.735,
}

# segment com positions as a function of segment length relative to proximal end
SEGMENT_COM_MALE = {
    "head": 59.76 / 100,
    "upper_trunk": 29.99 / 100,
    "mid_trunk": 45.02 / 100,
    "lower_trunk": 61.15 / 100,
    "upper_arm": 57.72 / 100,
    "forearm": 45.74 / 100,
    "hand": 79.00 / 100,
    "upper_leg": 40.95 / 100,
    "lower_leg": 44.59 / 100,
    "foot": 44.15 / 100,
}

SEGMENT_COM_FEMALE = {
    "head": 58.94 / 100,
    "upper_trunk": 20.77 / 100,
    "mid_trunk": 45.12 / 100,
    "lower_trunk": 49.20 / 100,
    "upper_arm": 57.54 / 100,
    "forearm": 45.59 / 100,
    "hand": 74.74 / 100,
    "upper_leg": 36.12 / 100,
    "lower_leg": 44.16 / 100,
    "foot": 40.14 / 100,
}

# Segment radius of gyration as a function of segment length
# sagittal, transverse, longitudinal
SEGMENT_RADII_GYRATION_MALE = {
    "head": asarray([36.2, 37.6, 31.2]) / 100,
    "upper_trunk": asarray([71.6, 45.4, 65.9]) / 100,
    "mid_trunk": asarray([48.2, 38.3, 46.8]) / 100,
    "lower_trunk": asarray([61.5, 55.1, 58.7]) / 100,
    "upper_arm": asarray([28.5, 26.9, 15.8]) / 100,
    "forearm": asarray([27.6, 26.5, 12.1]) / 100,
    "hand": asarray([62.8, 51.3, 40.1]) / 100,
    "upper_leg": asarray([32.9, 32.9, 14.9]) / 100,
    "lower_leg": asarray([25.5, 24.9, 10.3]) / 100,
    "foot": asarray([25.7, 24.5, 12.4]) / 100,
}

SEGMENT_RADII_GYRATION_FEMALE = {
    "head": asarray([33.0, 35.9, 31.8]) / 100,
    "upper_trunk": asarray([74.6, 50.2, 71.8]) / 100,
    "mid_trunk": asarray([43.3, 35.4, 41.5]) / 100,
    "lower_trunk": asarray([43.3, 40.2, 44.4]) / 100,
    "upper_arm": asarray([27.8, 26.0, 14.8]) / 100,
    "forearm": asarray([26.1, 25.7, 9.4]) / 100,
    "hand": asarray([53.1, 45.4, 33.5]) / 100,
    "upper_leg": asarray([36.9, 36.4, 16.2]) / 100,
    "lower_leg": asarray([27.1, 26.7, 9.3]) / 100,
    "foot": asarray([29.9, 27.9, 13.9]) / 100,
}

# Map segment to xsens
SEGMENT_TO_XSENS = {
    "head": [mvn.SEGMENT_HEAD, mvn.SEGMENT_NECK],
    "upper_trunk": [mvn.SEGMENT_T8],
    "mid_trunk": [mvn.SEGMENT_T12, mvn.SEGMENT_L3],
    "lower_trunk": [mvn.SEGMENT_L5, mvn.SEGMENT_PELVIS],
    "l_upper_arm": [mvn.SEGMENT_LEFT_UPPER_ARM],
    "r_upper_arm": [mvn.SEGMENT_RIGHT_UPPER_ARM],
    "l_forearm": [mvn.SEGMENT_LEFT_FOREARM],
    "r_forearm": [mvn.SEGMENT_RIGHT_FOREARM],
    "l_hand": [mvn.SEGMENT_LEFT_HAND],
    "r_hand": [mvn.SEGMENT_RIGHT_HAND],
    "l_upper_leg": [mvn.SEGMENT_LEFT_UPPER_LEG],
    "r_upper_leg": [mvn.SEGMENT_RIGHT_UPPER_LEG],
    "l_lower_leg": [mvn.SEGMENT_LEFT_LOWER_LEG],
    "r_lower_leg": [mvn.SEGMENT_RIGHT_LOWER_LEG],
    "l_foot": [mvn.SEGMENT_LEFT_FOOT, mvn.SEGMENT_LEFT_TOE],
    "r_foot": [mvn.SEGMENT_RIGHT_FOOT, mvn.SEGMENT_RIGHT_TOE],
}
# Helper functions
def strip_utf8(s):
    if s[0] == "\ufeff":
        s = s[1 : len(s)]
    return s


def strip_endline(s):
    if s[-1] == "\n":
        s = s[0:-1]
    return s


def configure_r():
    # for row in iter_info():
    #     print(row)
    utils = importr("utils")
    base = importr("base")
    stats = importr("stats")
    mgcv = importr("mgcv")

    return utils, base, stats, mgcv


def convert_to_g(x):
    return x / GRAVITY


# def skew(x):
#     """
#     Compute the matrix cross product operator (skew symmetric matrix) for
#     a vector x.
#     """
#     if len(x) != 3:
#         raise ValueError("x must be of length 3")

#     return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


# def exp_map(A):
#     """
#     Computes the exponential map of a skew symmetric matrix A via
#     the Euler-Rodrigues formula
#     """

#     # TODO check A skew symmetric
#     mat = np.identity(3)
#     fa = np.sqrt(np.trace((A @ np.transpose(A))) / 2)
#     mat += (np.sin(fa) / fa) * A + ((1 - np.cos(fa) / fa**2)) * A**2
#     return mat


# def quat_to_rot_mat(q):
#     """
#     Convert a quaternion to a rotation matrix
#     :q:
#     """
#     if len(q) != 4:
#         raise ValueError("q must be a quaternion of (length 4 vector...)")

#     rotMat = np.zeros((3, 3))
#     qr = q[0]
#     qi = q[1]
#     qj = q[2]
#     qk = q[3]
#     s = np.sqrt(np.dot(q, q))

#     rotMat[0, 0] = 1 - 2 * s * (qj**2 + qk**2)
#     rotMat[0, 1] = 2 * s * (qi * qj - qk * qr)
#     rotMat[0, 2] = 2 * s * (qi * qk + qj * qr)
#     rotMat[1, 0] = 2 * s * (qi * qj - qk * qr)
#     rotMat[1, 1] = 1 - 2 * s * (qi**2 + qk**2)
#     rotMat[1, 2] = 2 * s * (qj * qk - qi * qr)
#     rotMat[2, 0] = 2 * s * (qi * qk - qj * qr)
#     rotMat[2, 1] = 2 * s * (qj * qk + qi * qr)
#     rotMat[2, 2] = 1 - 2 * s * (qi**2 + qj**2)
#     return rotMat


# def rotate_quat(x, q):
#     """
#     Compute the hamilton prod to rotate vector x by quaternion q

#     :x: length 3 vector
#     :q: length 4 vector quaternion
#     :x': rotated x
#     """
#     if len(x) != 3:
#         raise ValueError("X must be a vector of length 3")
#     if len(q) != 4:
#         raise ValueError("q must be a quaternion of (length 4 vector...)")
#     q_inv = q
#     q_inv[1:] = -1 * q_inv[1:]
#     x = np.asarray(x)
#     x = asarray([0, *x])
#     return (q * x * q_inv)[1:]


def rel_quat(x, y):
    """
    Compute the quaternion describing how to rotate x to get y
    """

    # normalize length
    x = x
    y = y

    # q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);

    ijk = np.cross(y, x)
    a = np.sqrt(np.linalg.norm(x) ** 2 * np.linalg.norm(y) ** 2) + np.dot(x, y)
    q = np.array([a, *ijk])
    q = q / np.linalg.norm(q)
    return np.quaternion(*q)


class Polyn:
    """
    Simple class to fit and predict from polynomials
    """

    # TODO add standard errors/predcition errors
    def __init__(self, x, y, q, coefs=None, zero_intcpt=False):
        if coefs is not None:
            if len(coefs) != q + 1:
                raise ValueError(
                    "Inappropiate size of supplied coefficents.  Coefficients of length {clen} need to be provided".format(
                        clen=q + 1
                    )
                )

        # if x is None:
        # #     raise ValueError("x not supplied.")

        # # if y is not None:
        # #     if len(x) != len(y):
        # #         raise ValueError("length of y does not match length of x")

        self.x = x
        self.y = y
        self.q = q
        self.coefs = coefs

        if coefs is None:
            X = self.__createX(x)
            if zero_intcpt:
                self.y = y - np.mean(y)
                self.coefs = (
                    np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ self.y
                )
                # self.coefs = np.array([0, *np.flip(np.polyfit(self.x, self.y, q))[1:]])
            else:
                self.coefs = (
                    np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ self.y
                )

                # self.coefs = np.flip(np.polyfit(self.x, self.y, q))

    def __str__(self):
        return """ Polynomial:\n\t Order: {self.q}\n\t Coefs: {self.coefs}""".format(
            self=self
        )

    def __createX(self, x):
        if x is None:
            x = self.x
        x = np.asarray([x])
        if len(x) > 1:
            X = np.zeros((len(self.x), self.q + 1))
            for i in range(self.q + 1):
                X[:, i] = x**i
        else:
            X = np.zeros((self.q + 1,))
            for i in range(self.q + 1):
                X[i] = x**i
        return X

    def predict(self, x=None, dev=0):
        if dev < 0:
            raise ValueError("dev must be >0")
        if x is None:
            x = self.x
        x = np.asarray(x)

        if dev == 0:
            # From: https://codereview.stackexchange.com/questions/131722/algorithm-to-compute-the-n-th-derivative-of-a-polynomial-in-python

            return self.__createX(x) @ self.coefs
        else:
            old_q = copy.deepcopy(self.q)
            poly = copy.deepcopy(self.coefs)
            new_q = old_q - dev
            p = 1
            for k in range(2, dev + 1):
                p *= k
            poly = poly[:-dev]
            n = len(poly) - 1
            for i in range(len(poly)):
                poly[n - i] *= p
                p = p * (i + dev + 1) // (i + 1)

            if len(x) > 1:
                X = np.zeros((len(self.x), new_q + 1))
                for i in range(new_q + 1):
                    X[:, i] = x**i
            else:
                X = np.zeros((new_q + 1,))
                for i in range(new_q + 1):
                    X[i] = x**i

            return X @ np.asarray(poly)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return min(zs)


if __name__ == "__main__":
    # configure_r()
    print("You have run util")
    import numpy as np

    q1 = np.quaternion(1, 0, 0, 0)
    q2 = quaternion.from_euler_angles(0, 0, np.pi / 2)
    v = np.array([1, 0, 0])
    print(q2)
    print(quaternion.rotate_vectors(q2, v))
