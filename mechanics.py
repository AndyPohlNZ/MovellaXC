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
    dx = np.abs(np.cos(slope)*(com[0] - hand[0]) - np.sin(slope)*(com[1] - hand[1]))
    return dx

sub = Subject(6)
sub.load_trial("hard", "xsens")
frame = 29000

n = sub.mvnx.frame_count

p = sub.poses[29000]

x = []
y = []

for i in range(1, n):
    p0 = sub.gnss.pos[i-1,:][0:2] 
    p1 = sub.gnss.pos[i, :][0:2]
    x.append(np.linalg.norm(p1-p0))
    y.append(sub.gnss.pos[i,2])

x = np.array([0, *np.cumsum(x)])
y = np.array([sub.gnss.pos[0,2], *y])
time = sub.gnss.time

dslope = np.zeros((n, 2))
dslope[:, 0] = intrp.CubicSpline(time,x).derivative(1)(time)
dslope[:, 1] = intrp.CubicSpline(time,y).derivative(1)(time)

for i in range(n):
    dslope[i,:] = dslope[i,:]/np.linalg.norm(dslope[i,:])



slope_angle = np.arctan2(np.diff(x), np.diff(y))

slope_angle = [0, *slope_angle]

x_com = np.zeros(np.shape(x))
y_com = np.zeros(np.shape(y))
for i in range(n):
    x_com[i] = x[i] + sub.poses[i].get_com()[0]
    y_com[i] = y[i] + sub.poses[i].get_com()[2]


ax = plt.subplot()
ax.plot(x, y)
ax.plot(x_com, y_com, color = util.CMAP(2))
ax2 = ax.twinx()
ax2.plot(x, slope_angle, color=util.CMAP(1))
plt.show()

# Compute hand position relative to contact
r_hand = np.zeros((n,2))
l_hand = np.zeros((n,2))
contact = np.zeros((n,2))
for i in range(n):
    contact_raw =np.array([sub.poses[i].points["RightToe"] + sub.poses[i].points["LeftToe"]])/2
    contact_raw = np.array([contact_raw[0][0], contact_raw[0][2]])
    contact[i] = contact_raw
    com = [sub.poses[i].get_com()[0], sub.poses[i].get_com()[2]]
    r_hand_raw = [sub.poses[i].points["RightHand"][0], sub.poses[i].points["RightHand"][2]]
    l_hand_raw = [sub.poses[i].points["LeftHand"][0], sub.poses[i].points["LeftHand"][2]]
    r_hand[i] = r_hand_raw -contact[i]
    l_hand[i] = l_hand_raw - contact[i]


ax = plt.subplot()
ax.plot(time[0:20], l_hand[0:20, 0])
plt.show()

# Compute hand velocity
dl_hand = np.zeros((n,2))
dl_hand[:,0] = intrp.CubicSpline(time, l_hand[:,0]).derivative(1)(time)
dl_hand[:,1] = intrp.CubicSpline(time, l_hand[:,1]).derivative(1)(time)

dr_hand = np.zeros((n,2))
dr_hand[:,0] = intrp.CubicSpline(time, r_hand[:,0]).derivative(1)(time)
dr_hand[:,1] = intrp.CubicSpline(time, r_hand[:,1]).derivative(1)(time)

l_speed_thresh = np.zeros((n, ))
r_speed_thresh = np.zeros((n, ))

for i in range(n):


    if np.linalg.norm(dl_hand[i,:])> 1: # Filter to speed > 1m/s
        l_speed_thresh[i] = 1

    if np.linalg.norm(dr_hand[i,:])> 1:
        r_speed_thresh[i] = 1

    dl_hand[i,:] = dl_hand[i,:]/np.linalg.norm(dl_hand[i,:])
    dr_hand[i,:] = dr_hand[i,:]/np.linalg.norm(dr_hand[i,:])

   

# Compute tip position
l_pole_tip = sub.pole_length * dl_hand + l_hand
r_pole_tip = sub.pole_length * dr_hand + r_hand

# Compute tip penetration
dx_l = np.zeros((n,))
dx_r = np.zeros((n,))
dx_l2 = np.zeros((n,))
dx_r2 = np.zeros((n,))

for i in range(n):
    if (np.arctan2(dl_hand[i,0], dl_hand[i,1])< -np.pi/2) & (np.arctan2(dl_hand[i,0], dl_hand[i,1])> -np.pi): 
        dx_l[i] = np.linalg.norm(np.cross(l_pole_tip[i,:], dslope[i,:]))/np.linalg.norm(dslope[i,:])
    if (np.arctan2(dr_hand[i,0], dr_hand[i,1])< -np.pi/2) & (np.arctan2(dr_hand[i,0], dr_hand[i,1])> -np.pi): 
        dx_r[i] = np.linalg.norm(np.cross(r_pole_tip[i,:], dslope[i,:]))/np.linalg.norm(dslope[i,:])
    

ax = plt.subplot()
ax.plot(time,dx_l)
ax.plot(time,dx_r)
plt.show()

# compute pole force
# Max pole force = 430 N Stoggl et al. 2016
MAX_POLE_FORCE = 450 # combined left/right
k_gc = MAX_POLE_FORCE/np.max(np.max([np.vstack((dx_l, dx_r))]))
Fp_l = np.zeros((n, 2))
Fp_r = np.zeros((n,2))
for i in range(n):
    Fp_l[i,:] =  l_speed_thresh[i] * k_gc * dx_l[i] * - dl_hand[i,:]/np.linalg.norm(dl_hand[i, :])
    Fp_r[i,:] = r_speed_thresh[i]*k_gc * dx_r[i]* -dr_hand[i,:]/np.linalg.norm(dl_hand[i,:])

ax = plt.subplot()
ax.plot(time,Fp_l[:,0])
ax.plot(time, Fp_l[:,1])
plt.show()



# Compute SKI FORCE
dt = 1/sub.mvnx.frame_rate
# Compute ski position relative to com
r_ski = np.zeros((n,2))
l_ski = np.zeros((n,2))
com = np.zeros((n,2))
for i in range(n):
    com[i,:] = [sub.poses[i].get_com()[0], sub.poses[i].get_com()[2]]
    r_ski_raw = [sub.poses[i].points["RightToe"][0], sub.poses[i].points["RightToe"][2]]
    l_ski_raw = [sub.poses[i].points["LeftToe"][0], sub.poses[i].points["LeftToe"][2]]
    r_ski[i,:] = r_ski_raw -com[i,:]
    l_ski[i,:] = l_ski_raw - com[i,:]

# compute ski velocity
# Compute hand velocity
dl_ski = np.zeros((n,2))
dl_ski[:,0] = intrp.CubicSpline(time, l_ski[:,0]).derivative(1)(time)
dl_ski[:,1] = intrp.CubicSpline(time, l_ski[:,1]).derivative(1)(time)

dr_ski = np.zeros((n,2))
dr_ski[:,0] = intrp.CubicSpline(time, r_ski[:,0]).derivative(1)(time)
dr_ski[:,1] = intrp.CubicSpline(time, r_ski[:,1]).derivative(1)(time)


ax = plt.subplot()
ax.plot(time,dl_ski[:,0])
plt.show()

#  compute force
k_sc = 0.8*sub.get_total_mass()*util.GRAVITY/np.max(np.vstack((dl_ski, dr_ski)))/2
Fs_l = np.zeros((n, 2))
Fs_r = np.zeros((n,2))
for i in range(n):
    if (dl_ski[i,0] <0) & (dl_ski[i,1] < 0):
        Fs_l[i,:] =  k_sc * np.linalg.norm(dl_ski[i,:]) * -dl_ski[i,:]/np.linalg.norm(dl_ski[i,:]) 
    if (dr_ski[i,0] <0) & (dr_ski[i,1] < 0):
        Fs_r[i,:] = k_sc * np.linalg.norm(dr_ski[i,:])* -dr_ski[i,:]/np.linalg.norm(dr_ski[i,:]) 

ax = plt.subplot()
ax.plot(time, Fs_l[:, 0])
ax.plot(time, Fs_l[:, 1])

plt.show()


# Compute Aero Drag
C_d = 0.67

#compute deriative of position
dx = np.zeros((n, 2))
dx[:, 0] = intrp.CubicSpline(time, x_com).derivative(1)(time)
dx[:, 1] = intrp.CubicSpline(time, y_com).derivative(1)(time)
# compute velocity
v = np.linalg.norm(dx, axis = 1)

# get area and drag force
frontal_area = np.zeros((n, ))
F_aero = np.zeros((n,))
for i in range(n):
    frontal_area[i] = sub.poses[i].area
    F_aero[i] = 0.5*sub.weather.density*C_d*v[i]**2



ax = plt.subplot()
ax.plot(time, F_aero)
plt.show()
# Compute friction
mu = 0.07


# sum Forces in X/y





# Friction
F = mu * mg * 

ax = plt.subplot()
ax.plot(time, F_aero)
plt.show()






com = np.array([p.get_com()[0], p.get_com()[2]])
hand = np.array([p.points["RightHand"][0], p.points["LeftHand"][2]])

hand = hand-com
np.abs(-hand[2])

ax = plt.subplot()
ax.scatter(p.points["RightHand"][2])
plt.show()


def get_potential_energy(sub, frame=-1):
    print("STATUS: Estimating Potential energy")

    h0 = sub.gnss.pos[:,2]
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




if __name__ == "__main__":








    p.plot_pose()
    M = sub.get_total_mass()
    h = np.zeros((n,))
    v= np.zeros((n, 3))
    v2 = np.zeros((n))
    for frame in range(n):
        h[frame] = sub.poses[frame].get_com()[2]
        v[frame,:] = sub.poses[frame].get_com_vel()
        v2[frame] = np.dot(v[frame,:], v[frame,:])

    h = h-np.min(h)
    E_pot = M*util.GRAVITY*h
    E_kin = 0.5*M*v2
    E_tot = E_pot + E_kin

    ax = plt.subplot()
    ax.plot(sub.gnss.time, E_pot)
    ax.plot(sub.gnss.time, E_kin)
    ax.plot(sub.gnss.time, E_tot)
    plt.show()

    def detrend(x, y):
        ytmp = copy.deepcopy(y)
        ytmp = ytmp-np.mean(ytmp)
        X = np.ones((len(x), 2))
        X[:,1] = x
        Xp = np.transpose(X)
        beta = np.linalg.inv(Xp@X)@Xp@y
        return ytmp - (X@beta)
    


    # Get velocity of left ski
    pelvis = np.zeros((n,3))
    pelvis_vel = np.zeros((n,3))
    lski = np.zeros((n, 3))
    rski = np.zeros((n,3))
    ski = np.zeros((n, 3))
    lski_vel = np.zeros((n, 3))
    rski_vel = np.zeros((n, 3))
    lski_acc = np.zeros((n, 3))
    rski_acc = np.zeros((n, 3))
    
    lhand = np.zeros((n, 3))
    rhand = np.zeros((n,3))
    lhand_vel = np.zeros((n, 3))
    rhand_vel = np.zeros((n,3))
    lhand_ori = []
    rhand_ori = []
    lski_vel = np.zeros((n, 3))
    rski_vel = np.zeros((n, 3))
    lski_acc = np.zeros((n, 3))
    rski_acc = np.zeros((n, 3))
    ski_vel = np.zeros((n, 3))
    ski_acc = np.zeros((n, 3))

    com = np.zeros((n, 3))
    com_vel = np.zeros((n, 3))
    com_acc= np.zeros((n,3))
    for i in range(n):
        lski[i,:] = sub.poses[i].points["LeftToe"] - sub.poses[i].get_com()
        rski[i, :] = sub.poses[i].points["RightToe"]- sub.poses[i].get_com()
        lhand[i,:] = sub.poses[i].points["LeftHand"]- sub.poses[i].get_com()
        rhand[i, :] = sub.poses[i].points["RightHand"]- sub.poses[i].get_com()
        pelvis[i, :] = sub.poses[i].points["Pelvis"]- sub.poses[i].get_com()
        com[i,:] = sub.poses[i].get_com()
        com_vel[i,:] = sub.poses[i].get_com_vel()
        com_acc[i,:] = sub.poses[i].get_com_acc()
        ski[i, :] = (lski[i,:] + rski[i,:])/2

    for i in range(3):
        pelvis_vel[:,i] = intrp.CubicSpline(sub.gnss.time, pelvis[:,i]).derivative(1)(sub.gnss.time) 

        lski_vel[:, i] = intrp.CubicSpline(sub.gnss.time, lski[:,i]).derivative(1)(sub.gnss.time) - pelvis_vel[:,i]
        rski_vel[:, i] = intrp.CubicSpline(sub.gnss.time, rski[:,i]).derivative(1)(sub.gnss.time)- pelvis_vel[:,i]
        lski_acc[:, i] = intrp.CubicSpline(sub.gnss.time, lski[:,i]).derivative(2)(sub.gnss.time) 
        rski_acc[:, i] = intrp.CubicSpline(sub.gnss.time, rski[:,i]).derivative(2)(sub.gnss.time)
        lhand_vel[:, i] = intrp.CubicSpline(sub.gnss.time, lhand[:,i]).derivative(1)(sub.gnss.time)  - pelvis_vel[:,i]
        rhand_vel[:, i] = intrp.CubicSpline(sub.gnss.time, rhand[:,i]).derivative(1)(sub.gnss.time) - pelvis_vel[:,i]
        #com_vel[:,i] =  intrp.CubicSpline(sub.gnss.time, com[:,i]).derivative(1)(sub.gnss.time) 
        ski_vel[:,i] = (lski_vel[:,i] + rski_vel[:,i])/2 -pelvis_vel[:,i]
        ski_acc[:,i] = (lski_acc[:,i] + rski_acc[:,i])/2

    # pelvis_ori =[]
    # pelvis_vel2 = []
    # lski_vel2 = []
    # for i in range(n):
    #     pelvis_ori.append(util.rel_quat([1,0,0], pelvis_vel[i,:]))
    #     pelvis_vel2.append(quat.rotate_vectors(pelvis_ori[i], pelvis_vel[i,:]))
    #     lski_vel2.append(quat.rotate_vectors(pelvis_ori[i], lski_vel[i,:]))


    lhand_plant = lhand_vel[:,2]>0.1
    rhand_plant = rhand_vel[:,2]>0.1
    nlhand_plant = lhand_vel[:,2]<0.1
    nrhand_plant = rhand_vel[:,2]<0.1
    gliding = np.int64((np.sqrt(ski_vel[:, 0]**2+ ski_vel[:, 1]**2)<0.2) &  nrhand_plant &nlhand_plant)



    # slope angle
    horz_disp = np.zeros((n,))
    disp = np.cumsum(np.sqrt(np.linalg.norm(xy, axis=1))*1/sub.mvnx.frame_rate)
    height = sub.gnss.pos[:,2]
    pos = sub.gnss.pos- sub.gnss.pos[0,:]



    np.sqrt(np.dot(p1-p0, p1-p0))

    dx= sub.gnss.spline[0].bspline.integrate(sub.gnss.time[i-1], sub.gnss.time[i])
    dy = sub.gnss.spline[1].bspline.integrate(sub.gnss.time[i-1], sub.gnss.time[i])


    disp = []
    disp2 = []
    for i in range(1,n,1):
        p0 = sub.gnss.pos[(i-1),:]
        p1 = sub.gnss.pos[i,:]
        disp.append(np.sqrt(np.dot(p1-p0, p1-p0)))


    disp=np.cumsum([0, *disp])

    spl = intrp.CubicSpline(disp, sub.gnss.pos[:,2])
    slope = spl.derivative(1)(disp)

    slope_ang = np.arctan(slope)

    ax = plt.subplot()
    ax.plot(disp, sub.gnss.pos[:,2])
    ax2 = ax.twinx()
    ax2.plot(disp, slope_ang, color=util.CMAP(2))
    
    plt.show()

    # Compute friction 2d
    def compute_mu(a,v, cp, rho, area, theta):
        return (m*np.sqrt(np.dot(a,a)) - 
                m*util.GRAVITY - 
                0.5*cp*rho*area*np.dot(v,v))/(np.sqrt((m*util.GRAVITY * np.cos(theta))**2))
    
    def compute_mu(a, v, m, cd, area, theta ):
        return(
             (((((1/(2*m))*cd*area*v**2) + a*np.sin(theta))/util.GRAVITY)- np.sin(theta))/(np.cos(theta))
        )
    
    i = 53756
    mu = []
    mu_time = []
    m = sub.get_total_mass()
    cd = 0.7
    for i in range(n):
        if gliding[i]==1:
            print(i)
            a= np.sqrt(np.dot(sub.poses[i].get_com_acc(),sub.poses[i].get_com_acc()))
            v = np.sqrt(np.dot(sub.poses[i].get_com_vel(), sub.poses[i].get_com_vel()))
            area = sub.poses[i].area
            theta = slope_ang[i]
            mu.append(compute_mu(m, 
                                 , 
                                 0.7, sub.weather.density, 
                                 sub.poses[i].area, 
                                 sub.poses[i].get_com_vel(), 
                                 slope_ang[i]))
            mu_time.append(sub.gnss.time[i])
        
    mu = np.array(mu)





    ax.plot(pos[:, 0], pos[:,1])
    plt.show()

sub.gnss.

ax = plt.subplot()
# ax.plot(sub.gnss.time, sub.gnss.pos[:,0]- sub.gnss.pos[0, 0]),
ax.plot(sub.gnss.time, sub.gnss.pos[:,1])
plt.show()

sub.gnss.spline[0].bspline.integrate(sub.gnss.time[-1], sub.gnss.time[0]) +sub.gnss.spline[1].bspline.integrate(sub.gnss.time[-1], sub.gnss.time[0])


    ax = plt.subplot()
    ax.plot(sub.gnss.time, lhand_plant, color="red", alpha=0.5)
    ax.plot(sub.gnss.time, rhand_plant, color="green", alpha=0.5)
    ax.plot(sub.gnss.time, gliding, color="blue")
    ax2 = ax.twinx()
    ax2.plot(sub.gnss.time, sub.gnss.pos[:,2], color="black")
    plt.show()


    ax = plt.subplot()
    ax.plot(sub.gnss.time, lhand_vel[:,2])
    ax.plot(sub.gnss.time, rhand_vel[:,2])
    ax.plot(sub.gnss.time, np.repeat(0, n))
    ax2 = ax.twinx()
    ax2.plot(sub.gnss.time, sub.gnss.pos[:,2], color="black")
    plt.show()


    lpole = np.zeros((n,3))
    rpole=np.zeros((n,3))
    for i in tqdm(range(5000)):
        lhand_ori.append(util.rel_quat([1,0,0], lhand_vel[i,:])/np.linalg.norm(lhand_vel[i,:]))
        rhand_ori.append(util.rel_quat([1,0,0], rhand_vel[i,:])/np.linalg.norm(rhand_vel[i,:]))
        left_pole = quat.rotate_vectors(1/lhand_ori[i], [sub.pole_length, 0,0]) + lhand
        right_pole = quat.rotate_vectors(1/rhand_ori[i], [sub.pole_length, 0,0]) + rhand


    ax = plt.subplot()
    ax.plot(sub.gnss.time, lhand_vel[:,0])
    ax.plot(sub.gnss.time, lhand_vel[:,1])
    ax.plot(sub.gnss.time, lhand_vel[:,2])
    plt.show()

    ax = plt.subplot()
    ax.plot(sub.gnss.time, lhand_vel[:,0])
    ax.plot(sub.gnss.time, lhand_vel[:,1])
    ax.plot(sub.gnss.time, lhand_vel[:,2])
    plt.show()

    fig = plt.figure(0)
    ax = plt.axes(projection="3d")
    frame = 1800
    ax.scatter(*sub.poses[frame].points["RightHand"], color="green")
    ax.scatter(*sub.poses[frame].points["LeftHand"], color="red")
    ax.scatter(*sub.poses[frame].points["Head"], color="black")
    ax.scatter(*sub.poses[frame].points["Pelvis"], color="black")
    ax.add_artist(
        util.Arrow3D(
            [sub.poses[frame].points["Pelvis"][0], sub.poses[frame].points["Head"][0]],
            [sub.poses[frame].points["Pelvis"][1], sub.poses[frame].points["Head"][1]],
            [sub.poses[frame].points["Pelvis"][2], sub.poses[frame].points["Head"][2]],
            **dict(mutation_scale=30, arrowstyle="-|>", color="black"),
        )
    ) 

    ax.scatter(*sub.poses[frame].points["RightToe"], color="green")
    ax.scatter(*sub.poses[frame].points["LeftToe"], color="red")
    ax.add_artist(
        util.Arrow3D(
            [sub.poses[frame].points["RightHand"][0], right_pole[frame, 0]],
            [sub.poses[frame].points["RightHand"][1], right_pole[frame, 1]],
            [sub.poses[frame].points["RightHand"][2], right_pole[frame, 2]],
            **dict(mutation_scale=30, arrowstyle="-|>", color="green"),
        )
    ) 
    ax.add_artist(
        util.Arrow3D(
            [sub.poses[frame].points["LeftHand"][0], left_pole[frame, 0]],
            [sub.poses[frame].points["LeftHand"][1], left_pole[frame, 1]],
            [sub.poses[frame].points["LeftHand"][2], left_pole[frame, 2]],
            **dict(mutation_scale=30, arrowstyle="-|>", color="red"),
        )
    )     

    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.set_title("Poles")
    plt.show(block=True)  # show plot




    ski_acc_mag = np.sqrt(np.linalg.norm(ski_acc, axis=1))
    ski_vel_mag = np.sqrt(np.linalg.norm(ski_vel, axis=1))
    com_acc_mag = detrend(sub.gnss.time, np.sqrt(np.linalg.norm(com_acc, axis=1)))+util.GRAVITY
    com_vel_mag = np.sqrt(np.linalg.norm(com_vel, axis=1))
    
    gliding = np.int64((ski_acc_mag<4) & (com_vel_mag >1))
    lpole_plant = lpole[:,2]-lski[:,2]
    rpole_plant = np.int64(rpole[:,2]-rski[:,2])


    ax = plt.subplot()
    ax.plot(sub.gnss.time, lhand[:,1])
    plt.show()
    ax.plot(sub.gnss.time, ski_vel_mag, alpha =0.5)
    ax.plot(sub.gnss.time, ski_acc_mag, alpha=0.5)
    ax.plot(sub.gnss.time, com_vel_mag, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(sub.gnss.time, sub.gnss.pos[:,2], color="black")
    ax2.plot(sub.gnss.time, gliding, color="blue")
    ax2.plot(sub.gnss.time, lpole_plant, color="red")
    ax2.plot(sub.gnss.time, rpole_plant, color="green")

    ax.set_ylim((0,10))
    plt.show(block=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sub.gnss.time, lski[:, 0])
    ax1.plot(sub.gnss.time, rski[:, 0])
    ax1.plot(sub.gnss.time, ski[:, 0])

    ax2.plot(sub.gnss.time, lski[:, 1])
    ax2.plot(sub.gnss.time, rski[:, 1])
    ax2.plot(sub.gnss.time, ski[:, 1])

    ax3.plot(sub.gnss.time, lski[:, 2])
    ax3.plot(sub.gnss.time, rski[:, 2])
    ax3.plot(sub.gnss.time, ski[:, 2])
    plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sub.gnss.time, lski_vel[:, 0])
    ax1.plot(sub.gnss.time, rski_vel[:, 0])
    ax1.plot(sub.gnss.time, ski_vel[:, 0])
    ax1.plot(sub.gnss.time, com_vel[:,0], color='black')

    ax2.plot(sub.gnss.time, lski_vel[:, 1])
    ax2.plot(sub.gnss.time, rski_vel[:, 1])
    ax2.plot(sub.gnss.time, ski_vel[:, 1])
    ax2.plot(sub.gnss.time, com_vel[:,1], color='black')

    ax3.plot(sub.gnss.time, lski_vel[:, 2])
    ax3.plot(sub.gnss.time, rski_vel[:, 2])
    ax3.plot(sub.gnss.time, ski_vel[:, 2])
    ax3.plot(sub.gnss.time, com_vel[:,2], color='black')

    plt.show()



    # Find gliding regions
    eps = 0.01
    lski_vmag = np.zeros((n,))
    rski_vmag = np.zeros((n,))

    for i in range(n):
        lski_vmag[i] = np.linalg.norm(lski_vel[i,:] - com_vel[i,:])
        rski_vmag[i] = np.linalg.norm(rski_vel[i,:] - com_vel[i,:])

    lski_vmag = detrend(sub.gnss.time, lski_vmag)
    ddlski_vmag = 
    rski_vmag = detrend(sub.gnss.time, rski_vmag)

    
    detrend(sub.gnss.time, lski_vmag)


    ax = plt.subplot()
    ax.plot(list(range(n)), lski_vmag)
    ax.plot(list(range(n)), rski_vmag)
    ax.plot(list(range(n)), np.sum((lski_vmag, rski_vmag), axis=0)/2)
    ax.plot(list(range(n)), sub.gnss.pos[:,2], color="black")

    plt.show()



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sub.gnss.time, lski_vel[:, 0])
    ax1.plot(sub.gnss.time, rski_vel[:, 0])
    ax1.plot(sub.gnss.time, ski_vel[:, 0])
    ax1.plot(sub.gnss.time, com_vel[:,0], color='black')

    ax2.plot(sub.gnss.time, lski_vel[:, 1])
    ax2.plot(sub.gnss.time, rski_vel[:, 1])
    ax2.plot(sub.gnss.time, ski_vel[:, 1])
    ax2.plot(sub.gnss.time, com_vel[:,1], color='black')

    ax3.plot(sub.gnss.time, lski_vel[:, 2])
    ax3.plot(sub.gnss.time, rski_vel[:, 2])
    ax3.plot(sub.gnss.time, ski_vel[:, 2])
    ax3.plot(sub.gnss.time, com_vel[:,2], color='black')

    plt.show()


    
   

    n = sub.mvnx.frame_count
    n_seg = len(sub.segment_mdl)
    p0, v0, a0 = sub.kalman.predict(sub, 0)
    h0 = p0[:, 2]
    for i in range(n):  # set lowest point to be height 0
        h0[i] = h0[i] - np.min(h0)

    p_com, _ = sub.get_segment_pos()
    v_com, lws = sub.get_segment_vel(remove_drift=True)
    v_com_d, _ = sub.get_segment_vel(remove_drift=False)
    frames = list(range(0, sub.mvnx.frame_count))

    dim = 2
    ax = plt.subplot()
    ax.plot(frames, v_com[:, dim], color=util.CMAP(0))
    ax.plot(frames, v_com_d[:, dim], color=util.CMAP(1))
    ax.plot(frames, lws[:, dim], color="black", linestyle="dashed")
    ax.plot(frames, np.repeat(0, n), color=util.CMAP(2))
    ax2 = ax.twinx()
    ax2.plot(frames, p_com[:, dim], color=util.CMAP(0), alpha=0.2)
    plt.show(block=False)

    M = 0
    for seg in sub.segment_mdl:
        M += sub.segment_mdl[seg].mass

    E_pot = get_potential_energy(sub) / M
    E_kin, E_kin_lin, E_kin_rot = get_kinetic_energy(sub)
    E_kin = E_kin / M
    E_kin_lin = E_kin_lin / M
    E_kin_rot = E_kin_rot / M

    ax = plt.subplot()
    ax.plot(E_kin, color=util.CMAP(0))
    ax.plot(E_kin_lin, color=util.CMAP(0), linestyle="dotted")
    ax.plot(E_kin_rot, color=util.CMAP(0), linestyle="dashed")

    ax.plot(E_pot, color=util.CMAP(1))
    ax.plot(E_kin + E_pot, color=util.CMAP(2))
    ax2 = ax.twinx()
    ax2.plot(h0, color="black")
    ax2.set_ylabel("Track Elevation [m]")
    ax2.legend(["Track Elevation"], loc="upper right")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy [J/kg]")
    ax.legend(["E_kin", "E_kin_lin", "E_kin_rot", "E_pot", "E_Total"], loc="upper left")
    plt.savefig(
        util.DATA_DIR + "output/P{sub.id}_{sub.trial}_Energy.png".format(sub=sub)
    )
    plt.show()


# Friction

# find where skis and com are travelling at approx the same velocity

# Com velocity
pos_com = np.asarray(sub.mvnx.get_center_of_mass_pos())

vel_com = np.asarray(sub.mvnx.get_center_of_mass_vel())
acc_com = np.asarray(sub.mvnx.get_center_of_mass_acc())

vel_l_ski = np.asarray(sub.mvnx.get_segment_vel(mvn.SEGMENT_LEFT_TOE))
vel_r_ski = np.asarray(sub.mvnx.get_segment_vel(mvn.SEGMENT_RIGHT_TOE))
vel_avg_ski = np.zeros((n, 3))
vel_diff = np.zeros((n, 3))

vel_com_cal = np.zeros((n, 3))
com_vel2 = np.zeros((n, 3))
for i in range(n):
    vel_avg_ski[i, :] = (vel_l_ski[i, :] + vel_r_ski[i, :]) / 2
    vel_diff[i, :] = vel_com[i, :] - vel_avg_ski[i, :]
    for (j, seg) in enumerate(sub.segment_mdl.keys()):
        vseg = np.zeros((len(util.SEGMENT_TO_XSENS), 3))
        for (k, xseg) in enumerate(util.SEGMENT_TO_XSENS[seg]):
            vseg[k, :] = sub.mvnx.get_segment_vel(xseg, i)

        vavg = np.mean(vseg, axis=0)
        vel_com_cal[i, 0] += sub.segment_mdl[seg].mass * vavg[0]
        vel_com_cal[i, 1] += sub.segment_mdl[seg].mass * vavg[1]
        vel_com_cal[i, 2] += sub.segment_mdl[seg].mass * vavg[2]


ax = plt.subplot()
ax.plot(pos_com[:, 0], color=util.CMAP(0))
ax.plot(pos_com[:, 1], color=util.CMAP(1))
ax.plot(pos_com[:, 2], color=util.CMAP(2))
plt.show()

ax = plt.subplot()
ax.plot(vel_diff[:, 0])
ax.plot(vel_diff[:, 1])
ax.plot(vel_diff[:, 2])

plt.show()

ax = plt.subplot()
ax.plot(vel_com[:, 0])
ax.plot(vel_com[:, 1])
ax.plot(vel_com[:, 2])
ax.plot(vel_com_cal[:, 0], color=util.CMAP(3))
ax.plot(vel_com_cal[:, 1], color=util.CMAP(4))
ax.plot(vel_com_cal[:, 2], color=util.CMAP(5))
ax.legend(["X", "Y", "Z", "X_cal", "y_cal", "z_cal"])
plt.show()

ax = plt.subplot()
ax.plot(vel_com_cal[:, 0])
ax.plot(vel_com_cal[:, 1])
ax.plot(vel_com_cal[:, 2])
ax.legend(["X", "Y", "Z"])
plt.show()

plt.show()

ax = plt.subplot()
ax.plot(acc_com[:, 0])
ax.plot(acc_com[:, 1])
ax.plot(acc_com[:, 2])
ax.legend(["X", "Y", "Z"])
plt.show()

p0, v0, a0 = sub.kalman.predict(sub, 0)

ax = plt.subplot()
ax.plot(a0[:, 0])
ax.plot(a0[:, 1])
ax.plot(a0[:, 2])
ax.legend(["X", "Y", "Z"])
plt.show()

tmp = np.asarray(sub.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS))

ax = plt.subplot()
ax.plot(tmp[:, 0])
ax.plot(tmp[:, 1])
ax.plot(tmp[:, 2])
ax.plot(tmp[:, 3])

ax.legend(["q0", "q1", "q2", "q3"])
plt.show()
np.linalg.norm(tmp[1000, :])


ax = plt.subplot()


ax.plot(sub.gnss.X, sub.gnss.Y)
ax.scatter(
    np.mean(sub.gnss.X[idx0:idx1]),
    np.mean(sub.gnss.Y[idx0:idx1]),
    color=util.CMAP(1),
    s=20,
)
ax.scatter(
    np.mean(sub.gnss.X[idx2:-1]), np.mean(sub.gnss.Y[idx2:-1]), color=util.CMAP(2), s=20
)

print(
    "Starting elevation {e1} - finishing elevation{e2}".format(
        e1=np.mean(sub.gnss.Z[idx0:idx1]), e2=np.mean(sub.gnss.Z[idx2:-1])
    )
)
plt.show()

ax = plt.subplot()
ax.plot(sub.gnss.Z - np.min(sub.gnss.Z))
plt.show()
