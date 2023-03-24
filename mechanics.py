""" Compute mechancis for a given trial:
    Output = Total energy(t) = Kinetic energy(t) + potential energy(t)
    Total power(t) = upper body power(t) + lower body power(t)
    Friction(t)
    Aero drag(t)

    """
import sys

sys.path.insert(0, "srcPython")
import util
import mvn
from subject import Subject
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from matplotlib import use as plt_use
from pspline import PSpline
import scipy.interpolate as intrp
import quaternion as quat
from pose import Pose

plt_use("MacOSX")


def delta_x(com, slope, hand):
    dx = np.abs(np.cos(slope) * (com[0] - hand[0]) - np.sin(slope) * (com[1] - hand[1]))
    return dx


sub = Subject(6)
sub.load_trial("hard", "xsens")
frame = 29000

n = sub.mvnx.frame_count

p = sub.poses[29000]

x = []
y = []

for i in range(1, n):
    p0 = sub.gnss.pos[i - 1, :][0:2]
    p1 = sub.gnss.pos[i, :][0:2]
    x.append(np.linalg.norm(p1 - p0))
    y.append(sub.gnss.pos[i, 2])

x = np.array([0, *np.cumsum(x)])
y = np.array([sub.gnss.pos[0, 2], *y])
time = sub.gnss.time

nslope = np.zeros((n, 2))
nslope[:, 0] = intrp.CubicSpline(time, x).derivative(1)(time)
nslope[:, 1] = intrp.CubicSpline(time, y).derivative(1)(time)

for i in range(n):
    nslope[i, :] = nslope[i, :] / np.linalg.norm(nslope[i, :])


slope_angle = np.arctan2(np.diff(x), np.diff(y))

slope_angle = [0, *slope_angle]


ax = plt.subplot()
ax.plot(x, y)
ax2 = ax.twinx()
ax2.plot(x, dslope2[:, 1], color=util.CMAP(1))
# ax2.plot(x, slope_angle, color = util.CMAP(1))
plt.show()


x_com = np.zeros(np.shape(x))
y_com = np.zeros(np.shape(y))
for i in range(n):
    x_com[i] = x[i] + sub.poses[i].get_com()[0]
    y_com[i] = y[i] + sub.poses[i].get_com()[2]


ax = plt.subplot()
ax.plot(x, y)
ax.plot(x_com, y_com, color=util.CMAP(2))
ax2 = ax.twinx()
ax2.plot(x, slope_angle, color=util.CMAP(1))
plt.show()

# Compute hand position relative to contact
r_hand = np.zeros((n, 2))
l_hand = np.zeros((n, 2))
contact = np.zeros((n, 2))
for i in range(n):
    contact_raw = (
        np.array([sub.poses[i].points["RightToe"] + sub.poses[i].points["LeftToe"]]) / 2
    )
    contact_raw = np.array([contact_raw[0][0], contact_raw[0][2]])
    contact[i] = contact_raw
    com = [sub.poses[i].get_com()[0], sub.poses[i].get_com()[2]]
    r_hand_raw = [
        sub.poses[i].points["RightHand"][0],
        sub.poses[i].points["RightHand"][2],
    ]
    l_hand_raw = [
        sub.poses[i].points["LeftHand"][0],
        sub.poses[i].points["LeftHand"][2],
    ]
    r_hand[i] = r_hand_raw - contact[i]
    l_hand[i] = l_hand_raw - contact[i]


ax = plt.subplot()
ax.plot(time[0:20], l_hand[0:20, 0])
plt.show()

# Compute hand velocity
dl_hand = np.zeros((n, 2))
dl_hand[:, 0] = intrp.CubicSpline(time, l_hand[:, 0]).derivative(1)(time)
dl_hand[:, 1] = intrp.CubicSpline(time, l_hand[:, 1]).derivative(1)(time)

dr_hand = np.zeros((n, 2))
dr_hand[:, 0] = intrp.CubicSpline(time, r_hand[:, 0]).derivative(1)(time)
dr_hand[:, 1] = intrp.CubicSpline(time, r_hand[:, 1]).derivative(1)(time)

l_speed_thresh = np.zeros((n,))
r_speed_thresh = np.zeros((n,))

for i in range(n):

    if np.linalg.norm(dl_hand[i, :]) > 1:  # Filter to speed > 1m/s
        l_speed_thresh[i] = 1

    if np.linalg.norm(dr_hand[i, :]) > 1:
        r_speed_thresh[i] = 1

    dl_hand[i, :] = dl_hand[i, :] / np.linalg.norm(dl_hand[i, :])
    dr_hand[i, :] = dr_hand[i, :] / np.linalg.norm(dr_hand[i, :])


# Compute tip position
l_pole_tip = sub.pole_length * dl_hand + l_hand
r_pole_tip = sub.pole_length * dr_hand + r_hand

# Compute tip penetration
dx_l = np.zeros((n,))
dx_r = np.zeros((n,))
dx_l2 = np.zeros((n,))
dx_r2 = np.zeros((n,))

for i in range(n):
    if (np.arctan2(dl_hand[i, 0], dl_hand[i, 1]) < -np.pi / 2) & (
        np.arctan2(dl_hand[i, 0], dl_hand[i, 1]) > -np.pi
    ):
        dx_l[i] = np.linalg.norm(
            np.cross(l_pole_tip[i, :], dslope[i, :])
        ) / np.linalg.norm(dslope[i, :])
    if (np.arctan2(dr_hand[i, 0], dr_hand[i, 1]) < -np.pi / 2) & (
        np.arctan2(dr_hand[i, 0], dr_hand[i, 1]) > -np.pi
    ):
        dx_r[i] = np.linalg.norm(
            np.cross(r_pole_tip[i, :], dslope[i, :])
        ) / np.linalg.norm(dslope[i, :])


ax = plt.subplot()
ax.plot(time, dx_l)
ax.plot(time, dx_r)
plt.show()

# compute pole force
# Max pole force = 430 N Stoggl et al. 2016
MAX_POLE_FORCE = 450  # combined left/right
k_gc = MAX_POLE_FORCE / np.max(np.max([np.vstack((dx_l, dx_r))]))
Fp_l = np.zeros((n, 2))
Fp_r = np.zeros((n, 2))
for i in range(n):
    Fp_l[i, :] = (
        l_speed_thresh[i]
        * k_gc
        * dx_l[i]
        * -dl_hand[i, :]
        / np.linalg.norm(dl_hand[i, :])
    )
    Fp_r[i, :] = (
        r_speed_thresh[i]
        * k_gc
        * dx_r[i]
        * -dr_hand[i, :]
        / np.linalg.norm(dl_hand[i, :])
    )

ax = plt.subplot()
ax.plot(time, Fp_l[:, 0])
ax.plot(time, Fp_l[:, 1])
plt.show()


# Compute SKI FORCE
dt = 1 / sub.mvnx.frame_rate
# Compute ski position relative to com
r_ski = np.zeros((n, 2))
l_ski = np.zeros((n, 2))
com = np.zeros((n, 2))
for i in range(n):
    com[i, :] = [sub.poses[i].get_com()[0], sub.poses[i].get_com()[2]]
    r_ski_raw = [sub.poses[i].points["RightToe"][0], sub.poses[i].points["RightToe"][2]]
    l_ski_raw = [sub.poses[i].points["LeftToe"][0], sub.poses[i].points["LeftToe"][2]]
    r_ski[i, :] = r_ski_raw - com[i, :]
    l_ski[i, :] = l_ski_raw - com[i, :]

# compute ski velocity
# Compute hand velocity
dl_ski = np.zeros((n, 2))
dl_ski[:, 0] = intrp.CubicSpline(time, l_ski[:, 0]).derivative(1)(time)
dl_ski[:, 1] = intrp.CubicSpline(time, l_ski[:, 1]).derivative(1)(time)

dr_ski = np.zeros((n, 2))
dr_ski[:, 0] = intrp.CubicSpline(time, r_ski[:, 0]).derivative(1)(time)
dr_ski[:, 1] = intrp.CubicSpline(time, r_ski[:, 1]).derivative(1)(time)


ax = plt.subplot()
ax.plot(time, dl_ski[:, 0])
plt.show()

#  compute force
k_sc = (
    0.8 * sub.get_total_mass() * util.GRAVITY / np.max(np.vstack((dl_ski, dr_ski))) / 2
)
Fs_l = np.zeros((n, 2))
Fs_r = np.zeros((n, 2))
for i in range(n):
    if (dl_ski[i, 0] < 0) & (dl_ski[i, 1] < 0):
        Fs_l[i, :] = (
            k_sc
            * np.linalg.norm(dl_ski[i, :])
            * -dl_ski[i, :]
            / np.linalg.norm(dl_ski[i, :])
        )
    if (dr_ski[i, 0] < 0) & (dr_ski[i, 1] < 0):
        Fs_r[i, :] = (
            k_sc
            * np.linalg.norm(dr_ski[i, :])
            * -dr_ski[i, :]
            / np.linalg.norm(dr_ski[i, :])
        )

ax = plt.subplot()
ax.plot(time, Fs_l[:, 0])
ax.plot(time, Fs_l[:, 1])

plt.show()


# Compute Aero Drag
C_d = 0.67

# compute deriative of position
dx = np.zeros((n, 2))
dx[:, 0] = intrp.CubicSpline(time, x_com).derivative(1)(time)
dx[:, 1] = intrp.CubicSpline(time, y_com).derivative(1)(time)
# compute velocity
v = intrp.CubicSpline(time, x).derivative(1)(time)


# get area and drag force
frontal_area = np.zeros((n,))
F_aero = np.zeros((n, 2))
for i in range(n):
    frontal_area[i] = sub.poses[i].area
    F_aero[i, :] = 0.5 * sub.weather.density * C_d * v[i] ** 2 * frontal_area[i] - dx[
        i, :
    ] / np.linalg.norm(dx[i, :])

ax = plt.subplot()
ax.plot(time, F_aero[:, 0])
ax.plot(time, F_aero[:, 1])
ax2 = ax.twinx()
ax2.plot(time, frontal_area)


plt.show()

# sum Forces in X/y
slope_angle = np.pi / 2 - np.arctan2(nslope[:, 0], nslope[:, 1])


def rotMat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# Project into slope coordinate system
SumF_partial = np.zeros((n, 2))
SumF_partial[:, 0] = Fp_l[:, 0] + Fp_r[:, 0] + Fs_l[:, 0] + Fs_r[:, 0] + F_aero[:, 0]
SumF_partial[:, 1] = (
    Fp_l[:, 1]
    + Fp_r[:, 1]
    + Fs_l[:, 1]
    + Fs_r[:, 1]
    + F_aero[:, 1]
    - sub.get_total_mass() * util.GRAVITY
)

SumF_partial_slope = np.zeros((n, 2))
for i in range(n):
    SumF_partial_slope[i, :] = rotMat(slope_angle[i]) @ SumF_partial[i, :]


# Compute friction
mu = 0.06


def compute_forces_with_friction(SumF_partial_slope, slope_angle, mu):
    SumF = np.zeros((n, 2))
    friction = mu * SumF_partial_slope[:, 1]  # compute friction
    SumF_partial_slope[:, 0] += friction  # add to sum in slope CS
    for i in range(n):  # reorient back to GCS
        SumF[i] = np.transpose(rotMat(slope_angle[i])) @ SumF_partial_slope[i, :]

    return SumF


SumF = compute_forces_with_friction(SumF_partial_slope, slope_angle, mu)


# Compute Work
Wt = np.zeros((n,))
Ws = np.zeros((n,))
Wp = np.zeros((n,))
xy = np.column_stack((x, y))

Fs = Fs_l + Fs_r
Fp = Fp_l + Fp_r
for i in range(1, n):
    Wt[i] = np.dot(SumF[i, :], dx[i, :] * dt)
    Ws[i] = np.dot(Fs[i, :], dx[i, :] * dt)
    Wp[i] = np.dot(Fp[i, :], dx[i, :] * dt)

Power = intrp.CubicSpline(time, np.exp(Wp)).derivative(1)(time)
Ps = intrp.CubicSpline(time, np.exp(Wp)).derivative(1)(time)
Pp = intrp.CubicSpline(time, np.exp(Ws)).derivative(1)(time)

Wp_smooth = intrp.splrep(time, Wp, s=1.6 * sub.mvnx.frame_count)
Ws_smooth = intrp.splrep(time, Ws, s=2.5 * sub.mvnx.frame_count)

fig, (ax, ax2) = plt.subplots(1, 2)
ax.plot(time, intrp.BSpline(*Wp_smooth)(time))
ax.plot(time, Wp, color=util.CMAP(0), alpha=0.1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Work [J]")
ax.set_ylim((-4, 20))
ax.set_title("Work done by upper body")

ax2.plot(time, intrp.BSpline(*Ws_smooth)(time))
ax2.plot(time, Ws, color=util.CMAP(0), alpha=0.1)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Work [J]")
ax2.set_ylim((-3, 20))
ax2.set_title("Work done by lower body")
plt.show()


def get_potential_energy(sub, frame=-1):
    print("STATUS: Estimating Potential energy")

    h0 = sub.gnss.pos[:, 2]
    n = sub.mvnx.frame_count
    M = sub.get_total_mass()

    if frame == -1:
        # pelvis_height = np.asarray(sub.mvnx.get_segment_pos(mvn.SEGMENT_PELVIS))[
        #     :, 2
        # ]  # note x./y dimensions are both 0

        # Set minimum height to zero.
        h0 = h0 - np.min(h0)

        E_pot = np.zeros(n)
        com_lcs, _ = sub.get_segment_pos()
        for frame in range(n):
            E_pot[frame] += M * util.GRAVITY * (h0[frame] + com_lcs[frame, 2])
    else:
        # pelvis_height = np.asarray(sub.mvnx.get_segment_pos(mvn.SEGMENT_PELVIS, frame))[
        #     2
        # ]  # note x./y dimensions are both 0

        h0 = h0[frame] - h0[0]
        com_lcs, _ = sub.get_segment_pos(frame=frame)
        E_pot = M * util.GRAVITY * (h0 + com_lcs[2])

    return E_pot


def get_kinetic_energy(sub, frame=-1):
    print("STATUS: Estimating Kinetic energy")
    _, v0, _ = sub.kalman.predict(sub, 0)
    n = sub.mvnx.frame_count
    if frame == -1:
        E_kin_lin = np.zeros((n,))
        E_kin_rot = np.zeros((n,))
        E_kin = np.zeros((n,))
        for (i, seg) in enumerate(tqdm(sub.segment_mdl.keys())):
            print("STATUS: Processing segment: {}".format(seg))
            I = np.diag(sub.segment_mdl[seg].inertia)
            v_i = np.zeros((n, len(util.SEGMENT_TO_XSENS[seg]), 3))
            omega_i = np.zeros((n, len(util.SEGMENT_TO_XSENS[seg]), 3))
            for (j, xseg) in enumerate(util.SEGMENT_TO_XSENS[seg]):
                v_i[:, j, :], _ = sub.get_segment_vel(segment=xseg)
                omega_i[:, j, :] = sub.mvnx.get_segment_angular_vel(xseg)
            v_i = np.mean(v_i, axis=1)
            omega_i = np.mean(omega_i, axis=1)

            for frame in range(n):
                I_f = util.quat_to_rot_mat(sub.mvnx.get_segment_ori(0, frame)) @ I

                v = v0[frame, :] + v_i[frame, :]
                E_kin_lin[frame] += 0.5 * sub.segment_mdl[seg].mass * np.dot(v, v)
                E_kin_rot[frame] += (
                    0.5 * np.transpose(omega_i[frame, :]) @ I_f @ omega_i[frame, :]
                )
                E_kin[frame] = E_kin_lin[frame] + E_kin_rot[frame]

    else:
        # TODO FROM HERE FOR SINGLE FRAME
        for (i, seg) in enumerate(sub.segment_mdl.keys()):
            I = np.diag(sub.segment_mdl[seg].inertia)
            v_i = np.zeros((len(util.SEGMENT_TO_XSENS[seg]), 3))
            omega_i = np.zeros((len(util.SEGMENT_TO_XSENS[seg]), 3))
            for (j, xseg) in enumerate(util.SEGMENT_TO_XSENS[seg]):
                v_i[j, :], _ = sub.get_segment_vel(xseg, frame)
                omega_i[j, :] = sub.mvnx.get_segment_angular_vel(xseg, frame)
            v_i = np.mean(v_i, axis=0)
            omega_i = np.mean(omega_i, axis=0)

            v = v0[frame, :] + v_i
            E_kin_lin = 0.5 * sub.segment_mdl[seg].mass * np.dot(v, v)
            E_kin_rot = 0.5 * np.transpose(omega_i) @ I @ omega_i
        E_kin = E_kin_lin + E_kin_rot

    return E_kin, E_kin_lin, E_kin_rot
