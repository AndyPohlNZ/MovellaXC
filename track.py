import util
import numpy as np
from pyproj import Proj
from rpy2.robjects import FloatVector, DataFrame, BoolVector
from rpy2.robjects import Formula as r_Formula
import rpy2.robjects as robjs
from pspline import PSpline
import copy
import matplotlib.pyplot as plt
import quaternion as quat
import open3d as o3d  # To Do minimal import
from tqdm import tqdm
import scipy.interpolate as intrp


TURN_KEYPOINTS = np.array(
    [
        [5128525.5, 205588.5],
        [5128484.4, 205553.4],
        [5128328.9, 205744.1],
        [5128356.0, 205771.5],
    ]
)


class Track:
    """
    Defines a track coordinate dataset
    """

    def __init__(self, filename, coord_sys):
        self.filename = filename
        self.utm_zone = 32  # note fixed to zone 32...
        self.mag_dec = 2.77  # magnetic declination
        self.raw_data = None
        self.raw_data_file_type = None
        self.coord_sys = coord_sys  # WGS84, UTM, MAG
        self.elevation_mdl = None

    def __plot_track(self, x, y, title=""):
        ax = plt.subplot()
        ax.plot(x, y)
        ax.set_title(title)
        plt.show(block=True)

    def __compute_convergence(self, long, lat):
        cent_meridian = (self.utm_zone - 1.0) * 6.0 - 180.0 + 3.0
        return (
            np.arctan(
                np.tan((long - cent_meridian) * np.pi / 180) * np.sin(lat * np.pi / 180)
            )
            * 180
            / np.pi
        )

    def __ellipse_vol(self, a, b, c):
        return 4 / 3 * np.pi * a * b * c

    def __set_coordinate_system(self, long, lat, height, coord_sys="wgs84"):
        print(
            "STATUS: Converting track to {coord_sys} coordinates".format(
                coord_sys=coord_sys
            )
        )

        xyz = np.zeros((len(long), 3))
        if coord_sys == "utm":
            p = Proj(proj="utm", zone=self.utm_zone, ellps="WGS84", preserve_units=True)
            for i in range(len(long)):
                est, nth = p(long, lat)
                xyz[i, :] = [est, nth, height[i]]
        elif self.coord_sys == "xsens":
            p = Proj(proj="utm", zone=self.utm_zone, ellps="WGS84", preserve_units=True)
            est = np.zeros(len(long))
            nth = np.zeros(len(long))
            for i in range(len(long)):
                est[i], nth[i] = p(long[i], lat[i])
            for i in range(len(long)):
                convergence = self.__compute_convergence(long[i], lat[i])
                if (convergence > 0) & (self.mag_dec > 0):
                    if convergence < self.mag_dec:
                        offset = (self.mag_dec - convergence) * np.pi / 180
                    else:
                        offset = (convergence - self.mag_dec) * np.pi / 180
                else:
                    raise ValueError("TODO when convergence or self.mag_dec <0")

                correction = np.matrix(
                    [
                        [np.cos(offset), -np.sin(offset)],
                        [np.sin(offset), np.cos(offset)],
                    ]
                )
                xy = np.squeeze(
                    correction
                    * np.matrix([[est[i] - np.mean(est)], [nth[i] - np.mean(nth)]])
                )
                xyz[i, 0] = xy[0, 1] + np.mean(nth)
                xyz[i, 1] = xy[0, 0] + np.mean(
                    est
                )  # This converst east to west util.MAX_EASTING - (xy[0, 0] + np.mean(est))
                # xyz[i, 0:2] = np.transpose(
                #     np.array([[0, 1], [1, 0]]) @ np.transpose(xyz[i, 0:2])
                # )  # Flip in xy line
                xyz[i, 2] = height[i]
        elif self.coord_sys == "wgs84":
            print("STATUS: Converting track to wgs84 coordinates")
            for i in range(len(long)):
                xyz[i, :] = np.array([long[i], lat[i], height[i]])

        return xyz[:, 0], xyz[:, 1], xyz[:, 2]

    def __train_elevation_model(self, coord_sys, std_err=False):
        if coord_sys == "xsens":
            file_name = util.DATA_DIR + "GPS RTK data/combined_coord_xsens.csv"
            data = np.genfromtxt(file_name, delimiter=",", skip_header=1)

            X = data[:, 0]
            Y = data[:, 1]
            Z = data[:, 2]

            weights = np.zeros(len(X))
            for i in range(len(X)):
                weights[i] = 1 / self.__ellipse_vol(data[i, 3], data[i, 3], data[i, 4])
            weights = weights / np.mean(weights)

        utils_r, base_r, stats_r, mgcv_r = util.configure_r()

        robjs.globalenv["X"] = FloatVector(X)
        robjs.globalenv["Y"] = FloatVector(Y)
        robjs.globalenv["Z"] = FloatVector(Z - np.mean(Z))
        robjs.globalenv["weights"] = FloatVector(weights)

        print("STATUS: Fitting elevation model in R")
        m0 = mgcv_r.gam(r_Formula("Z ~ s(X, Y ,k=30)"), weights=FloatVector(weights))
        # print("  MGCV Elevation Model:")
        # print(base_r.summary(m0))

        return m0

    def pred_elevation(self, mdl, X, Y, std_err=False):
        _, _, stats_r, _ = util.configure_r()
        if std_err:
            raise ValueError("Std Error not currently working")
        else:
            dataframe = DataFrame({"X": FloatVector(X), "Y": FloatVector(Y)})
            pred = np.asarray(stats_r.predict(mdl, newdata=dataframe))
            # pred  =stats_r['predict']
        return pred

    def load_track_data(self, file_type="rtk", coord_sys="wgs84"):
        if file_type not in ["rtk", "gnss"]:
            raise ValueError("{file_type} not supported.".format(file_type=file_type))

        if coord_sys not in ["wgs84", "utm", "xsens"]:
            raise ValueError("{coord_sys} not supported.".format(coord_sys=coord_sys))

        self.raw_data_file_type = file_type
        self.coord_sys = coord_sys

        if file_type == "rtk":
            print("STATUS: Loading track from rtk file")
            cols_of_interest = (
                1,  # Time
                13,  # Long
                14,  # Lat
                15,  # Height above geoid
                16,  # Height MSL
                17,  # Horz Acc
                18,  # Vert Acc
            )
            if self.filename[-12:-10] == "P3":
                # trim first doggy lap...
                tmp_data = np.genfromtxt(
                    self.filename,
                    delimiter=",",
                    skip_header=1141,
                    usecols=cols_of_interest,
                )
            else:
                tmp_data = np.genfromtxt(
                    self.filename,
                    delimiter=",",
                    skip_header=1,
                    usecols=cols_of_interest,
                )
            nrow, _ = tmp_data.shape
            tmp_data[:, 0] = np.arange(
                0, nrow / 4, 0.25
            )  # convert frame number to time given recording rate of 4Hz.
            tmp_data[:, 1] = tmp_data[:, 1] / 1e7  # Long convert to deg
            tmp_data[:, 2] = tmp_data[:, 2] / 1e7  # Lat: convert to deg
            tmp_data[:, 3] = tmp_data[:, 3] / 1e3  # Height ellopsid convert to m
            tmp_data[:, 4] = tmp_data[:, 4] / 1e3  # Height MSL convert to m
            tmp_data[:, 5] = tmp_data[:, 5] / 1e5  # Horz Acc convert to m
            tmp_data[:, 6] = tmp_data[:, 6] / 1e5  # Vert Acc convert to m
            X, Y, Z = self.__set_coordinate_system(
                tmp_data[:, 1], tmp_data[:, 2], tmp_data[:, 4], coord_sys
            )

            self.rtk_pos = np.column_stack((X, Y, Z))

            self.horz_acc = tmp_data[:, 5]
            self.vert_acc = tmp_data[:, 5]

        elif file_type == "gnss":
            print("STATUS: Loading track from gnss file")
            tmp_data = np.genfromtxt(self.filename, delimiter=",", skip_header=1)
            nrow, _ = tmp_data.shape
            tmp_data[:, 0] = np.arange(
                0, nrow / 240, 1 / 240
            )  # convert frame number to time given recording rate of 240Hz.

            # Cols: frame, Lat, Long Alt
            gnss_xyz = np.zeros((nrow, 3))
            if self.coord_sys == "utm":
                print("STATUS: Converting track to utm coordinates")
                p = Proj(
                    proj="utm", zone=self.utm_zone, ellps="WGS84", preserve_units=True
                )
                for i in range(nrow):
                    est, nth = p(tmp_data[i, 2], tmp_data[i, 1])
                    gnss_xyz[i, 0:1] = np.array([est, nth])

            elif self.coord_sys == "xsens":
                self.elevation_mdl = self.__train_elevation_model(coord_sys)

                print("STATUS: Converting track to xsens coordinates")
                # print(self.elevation_mdl)
                p = Proj(
                    proj="utm", zone=self.utm_zone, ellps="WGS84", preserve_units=True
                )
                est = np.zeros(nrow)
                nth = np.zeros(nrow)
                for i in range(nrow):
                    est[i], nth[i] = p(tmp_data[i, 2], tmp_data[i, 1])

                for i in range(nrow):
                    #     print("Long = {long} lat = {lat}".format(long = tmp_data[i, 1], lat = tmp_data[i,2]))
                    convergence = self.__compute_convergence(
                        tmp_data[i, 1], tmp_data[i, 2]
                    )
                    #     print("Convergence = {convergence}".format(convergence=convergence))
                    if (convergence > 0) & (self.mag_dec > 0):
                        if convergence < self.mag_dec:
                            offset = (self.mag_dec - convergence) * np.pi / 180
                        else:
                            offset = (convergence - self.mag_dec) * np.pi / 180
                    else:
                        raise ValueError("TODO when convergence or self.mag_dec <0")

                    correction = np.matrix(
                        [
                            [np.cos(offset), -np.sin(offset)],
                            [np.sin(offset), np.cos(offset)],
                        ]
                    )
                    # TODO check this still works without zero offset
                    xy = np.squeeze(
                        correction
                        * np.matrix([[est[i] - np.mean(est)], [nth[i] - np.mean(nth)]])
                    )
                    gnss_xyz[i, 0] = xy[0, 1] + np.mean(nth)
                    # TODO CHECK xy[0,0??]
                    gnss_xyz[i, 1] = xy[0, 0] + np.mean(
                        est
                    )  # THIS CONVERTS TO West north up where we want east north uputil.MAX_EASTING - (xy[0, 0] + np.mean(est))
                    # gnss_xyz[i, 0:2] = np.transpose(
                    #     np.array([[0, 1], [1, 0]]) @ np.transpose(gnss_xyz[i, 0:2])
                    # )
            elif self.coord_sys == "wgs84":
                print("STATUS: Converting to wgs84 coordinates")
                for i in range(nrow):
                    gnss_xyz[i, 0] = tmp_data[i, 2]
                    gnss_xyz[i, 1] = tmp_data[i, 1]

            self.__smooth_track(
                gnss_xyz[:, 0],
                gnss_xyz[:, 1],
                self.pred_elevation(self.elevation_mdl, gnss_xyz[:, 0], gnss_xyz[:, 1]),
                tmp_data[:, 0],
            )
            self.__in_corner()
            norms = self.estimate_normals()
            self.surface_normal = norms

        return self

    def __smooth_track(self, x, y, z, times, bDeg=5, pDeg=3):
        """Smooth track via a 4th order spline with 3rd order penalty"""
        print("STATUS: Smoothing track data with a {} degree P-Spline".format(bDeg))

        xyz2 = np.column_stack((x, y, z))

        frame_idx = np.array(list(range(xyz2.shape[0])))
        xyz_u, idx = np.unique(xyz2, return_index=True, axis=0)
        frame_idx = frame_idx[idx]
        # NOTE np.unique changes row order....
        order_idx = np.argsort(frame_idx)
        xyz_u = xyz_u[order_idx, :]
        frame_idx = frame_idx[order_idx]
        times_u = times[frame_idx]

        spls = []

        pos = np.zeros((len(times), 3))
        vel = np.zeros((len(times), 3))
        acc = np.zeros((len(times), 3))

        origin = xyz_u[0, :]
        for i in range(3):
            spls.append(
                PSpline(
                    times_u,
                    xyz_u[:, i] - origin[i],
                    nK=int(len(times_u) / 2),
                    bDeg=bDeg,
                    pDeg=pDeg,
                    pad=10,
                )
            )
            pos[:, i] = spls[i].bspline(times) + origin[i]
            vel[:, i] = spls[i].bspline.derivative(1)(times)
            acc[:, i] = spls[i].bspline.derivative(2)(times)

        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.time = times
        self.spline = spls

    def __two_pt_line(self, a, b):
        deltay = b[1] - a[1]
        deltax = b[0] - a[0]
        return util.Polyn(
            x=None, y=None, q=1, coefs=[a[1] - deltay / deltax * a[0], deltay / deltax]
        )

    def __in_corner(self):
        line_a = self.__two_pt_line(TURN_KEYPOINTS[0, :], TURN_KEYPOINTS[1, :])
        line_b = self.__two_pt_line(TURN_KEYPOINTS[2, :], TURN_KEYPOINTS[3, :])
        out = np.zeros((len(self.pos[:, 0]),))
        for i in range(self.pos.shape[0]):
            if (self.pos[i, 1] < line_a.predict(self.pos[i, 0])) | (
                self.pos[i, 1] > line_b.predict(self.pos[i, 0])
            ):
                out[i] = 1
        self.corner_status = out

    def estimate_normals(self, algo="crossprod", npts=8, radius=2.5):

        if algo == "randpts":
            print("STATUS: Estimating Surface Normals via random points")

            tmp_normals = np.zeros((len([*range(0, len(self.time), 240)]), 3))
            tmp_times = []
            idx = 0
            for frame in tqdm(range(0, len(self.time), 240)):
                vecs = np.zeros((npts, 3))
                p0 = self.pos[frame, :]
                for i in range(npts):
                    alpha = 2 * np.pi * np.random.random()  # angle
                    r = radius * np.random.random()  # random radius
                    vecs[i, 0:2] = np.array(
                        [r * np.cos(alpha) + p0[0], r * np.sin(alpha) + p0[1]]
                    )
                vecs[:, 2] = self.pred_elevation(
                    self.elevation_mdl, vecs[:, 0], vecs[:, 1]
                )
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(vecs)
                source.paint_uniform_color((1, 0, 0))
                tmp = np.zeros((npts, 3))
                tmp[:, 2] = 1.0
                source.normals = o3d.utility.Vector3dVector(tmp)
                source.estimate_normals()
                tmp_normals[idx, :] = np.mean(np.array(source.normals), axis=0)
                tmp_normals[idx, :] = tmp_normals[idx, :] / np.linalg.norm(
                    tmp_normals[idx, :]
                )
                tmp_times.append(self.time[frame])
                idx += 1

            # fit spline to normals
            print("STATUS: smoothing normals with a cubic spline")
            out = np.zeros((len(self.time), 3))
            for i in range(3):
                spl = intrp.CubicSpline(tmp_times, tmp_normals[:, i])
                out[:, i] = spl(self.time)
        else:
            print("STATUS: estiamting normals via simple cross product")
            tmp_out = np.zeros((len(self.time), 3))
            pa = copy.deepcopy(self.pos) + [1, 0, 0]
            pb = copy.deepcopy(self.pos) + [0, 1, 0]
            pa[:, 2] = self.pred_elevation(self.elevation_mdl, pa[:, 0], pa[:, 1])
            pb[:, 2] = self.pred_elevation(self.elevation_mdl, pb[:, 0], pb[:, 1])
            for frame in tqdm(range(len(self.time))):
                tmp_norm = np.cross(
                    pa[frame, :] - self.pos[frame, :], pb[frame, :] - self.pos[frame, :]
                )
                tmp_out[frame, :] = tmp_norm / np.linalg.norm(tmp_norm)

            out = np.zeros((len(self.time), 3))
            spls = []
            for i in range(3):
                print(i)
                spls.append(intrp.CubicSpline(self.time, tmp_out[:, i]))
                # PSpline(
                #     self.time,
                #     tmp_out[:, i],
                #     nK=int(tmp_out.shape[0] / 4),
                #     bDeg=3,
                #     pDeg=2,
                #     pad=10,
                # )

                out[:, i] = spls[i](self.time)

        return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "srcPython")
    import util
    from subject import Subject
    import numpy as np
    import matplotlib.pyplot as plt
    from pspline import PSpline

    # sub = Subject(5)
    # sub.load_trial("hard", "xsens")

    # ax = plt.subplot()
    # ax.plot(sub.gnss.pos[:, 0], sub.gnss.pos[:, 1])
    # plt.show()

    # f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # cols = [
    #     util.CMAP(sub.gnss.corner_status[i]) for i in range(len(sub.gnss.corner_status))
    # ]
    # # for i in range(len(sub.gnss.corner_status)):
    # # ax.scatter(sub.gnss.pos[:, 0], sub.gnss.pos[:, 1], color=cols[i])
    # ax1.plot(sub.gnss.time, sub.gnss.pos[:, 0])
    # ax1_2 = ax1.twinx()
    # ax1_2.plot(sub.gnss.time, sub.gnss.vel[:, 0], color=util.CMAP(1))
    # ax1_2.plot(sub.gnss.time, sub.gnss.acc[:, 0], color=util.CMAP(2))
    # ax1.set_xlabel("Time(s)")
    # ax1.set_ylabel("X Pos(m)")
    # ax1_2.set_ylabel("X vel/acc")

    # ax2.plot(sub.gnss.time, sub.gnss.pos[:, 1])
    # ax2_2 = ax2.twinx()
    # ax2_2.plot(sub.gnss.time, sub.gnss.vel[:, 1], color=util.CMAP(1))
    # ax2_2.plot(sub.gnss.time, sub.gnss.acc[:, 1], color=util.CMAP(2))
    # ax2.set_xlabel("Time(s)")
    # ax2.set_ylabel("Y Pos(m)")
    # ax2_2.set_ylabel("Y vel/acc")

    # ax3.plot(sub.gnss.time, sub.gnss.pos[:, 2])
    # ax3_2 = ax3.twinx()
    # ax3_2.plot(sub.gnss.time, sub.gnss.vel[:, 2], color=util.CMAP(1))
    # ax3_2.plot(sub.gnss.time, sub.gnss.acc[:, 2], color=util.CMAP(2))
    # ax3.set_xlabel("Time(s)")
    # ax3.set_ylabel("Z Pos(m)")
    # ax3_2.set_ylabel("XZvel/acc")

    # ax1_2.legend(["Vel", "Acc"])

    # plt.show(block=True)
    # # euler_angs = np.zeros((sub.mvnx.frame_count, 3))
    # # for frame in range(sub.mvnx.frame_count):
    # #     q = util.rel_quat(sub.gnss.surface_normal[frame, :], [1, 0, 0])
    # #     euler_angs[frame, :] = quat.as_euler_angles(q)
    # ax = plt.subplot()
    # ax.plot(sub.gnss.time, sub.gnss.surface_normal[:, 0], color=util.CMAP(0))
    # ax.plot(sub.gnss.time, sub.gnss.surface_normal[:, 1], color=util.CMAP(1))
    # ax.plot(sub.gnss.time, sub.gnss.surface_normal[:, 2], color=util.CMAP(2))
    # ax2 = ax.twinx()
    # ax2.plot(sub.gnss.time, sub.gnss.pos[:, 2], color="black")
    # ax.set_title("Plot of surface_normals")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Normals")
    # ax2.set_ylabel("Elevation [m]")
    # ax.legend(["X", "Y", "Z"])
    # plt.show()

    # pt = np.array([5128521.0, 20554.1])
    # pt = np.array([5128406.5, 205655.7])
    # pt = np.array([5128327.6, 205766.7])

    # Get unique values (given lower sampling rate for gnss sensor)
    # note values are normalized such that the first observation is point 0,0,0
    # xyz = np.column_stack(
    #     (
    #         sub.gnss.X - sub.gnss.X[0],
    #         sub.gnss.Y - sub.gnss.Y[0],
    #         sub.gnss.Z - sub.gnss.Z[0],
    #     )
    # )

    # frame_idx = np.array(list(range(xyz.shape[0])))
    # xyz_u, idx = np.unique(xyz, return_index=True, axis=0)
    # frame_idx = frame_idx[idx]
    # # NOTE np.unique changes row order....
    # order_idx = np.argsort(frame_idx)
    # xyz_u = xyz_u[order_idx, :]
    # frame_idx = frame_idx[order_idx]
    # times = sub.gnss.time[frame_idx]
    # xyz = copy.deepcopy(xyz_u)
    # spls = []
    # for i in range(3):
    #     spls.append(PSpline(times, xyz[:, i], 4, 3, pad=10))

    # ax = plt.subplot()
    # ax.scatter(times, xyz[:, 0], color=util.CMAP(0))
    # ax.plot(sub.gnss.time, spls[0].bspline(sub.gnss.time), color=util.CMAP(0))
    # ax.plot(
    #     sub.gnss.time,
    #     spls[0].bspline.derivative(1)(sub.gnss.time),
    #     color=util.CMAP(0),
    #     linestyle="dashed",
    # )
    # ax.plot(
    #     sub.gnss.time,
    #     spls[0].bspline.derivative(2)(sub.gnss.time),
    #     color=util.CMAP(0),
    #     linestyle="dotted",
    # )

    # ax.scatter(times, xyz[:, 1], color=util.CMAP(1))
    # ax.plot(sub.gnss.time, spls[1].bspline(sub.gnss.time), color=util.CMAP(1))
    # ax.plot(
    #     sub.gnss.time,
    #     spls[1].bspline.derivative(1)(sub.gnss.time),
    #     color=util.CMAP(1),
    #     linestyle="dashed",
    # )
    # ax.plot(
    #     sub.gnss.time,
    #     spls[1].bspline.derivative(2)(sub.gnss.time),
    #     color=util.CMAP(1),
    #     linestyle="dotted",
    # )

    # ax.scatter(times, xyz[:, 2], color=util.CMAP(2))
    # ax.plot(sub.gnss.time, spls[2].bspline(sub.gnss.time), color=util.CMAP(2))
    # ax.plot(
    #     sub.gnss.time,
    #     spls[2].bspline.derivative(1)(sub.gnss.time),
    #     color=util.CMAP(2),
    #     linestyle="dashed",
    # )
    # ax.plot(
    #     sub.gnss.time,
    #     spls[2].bspline.derivative(2)(sub.gnss.time),
    #     color=util.CMAP(2),
    #     linestyle="dotted",
    # )

    # plt.show()

    filename1 = util.DATA_DIR + "GPS RTK data/P3_coord.csv"
    track1 = Track(filename1, "xsens")
    track1.load_track_data("rtk", coord_sys="xsens")
    print(track1)
    track1.rtk_pos = track1.rtk_pos[
        1141:,
    ]
    track1.rtk_pos[:, 2] = track1.rtk_pos[:, 2] - np.mean(track1.rtk_pos[:, 2])
    # track1.Y = track1.Y[1141 : len(track1.Y)]
    # track1.Z = track1.Z[1141 : len(track1.Z)]
    # track1.Z = track1.Z - np.mean(track1.Z)
    track1.horz_acc = track1.horz_acc[1141 : len(track1.horz_acc)]
    track1.vert_acc = track1.vert_acc[1141 : len(track1.vert_acc)]

    track1_comb = np.column_stack((track1.rtk_pos, track1.horz_acc, track1.vert_acc))
    # print(track1.X)
    # print(track1.Y)
    # print(track1.Z)

    # ax = plt.subplot()
    # ax.plot(track1.rtk_pos[:, 1], track1.rtk_pos[:,0])
    # ax.plot(track2.rtk_pos[:, 1], track2.rtk_pos[:,0])

    # plt.show()

    # ax = plt.subplot()
    # ax.plot(track1.rtk_pos[:, 1], track1.rtk_pos[:,2])
    # ax.plot(track2.rtk_pos[:, 1], track2.rtk_pos[:,2])

    # plt.show()

    filename2 = util.DATA_DIR + "GPS RTK data/P13_coord.csv"
    track2 = Track(filename2, "xsens")
    track2.load_track_data("rtk", coord_sys="xsens")
    # track2.Z = track2.Z - np.mean(track2.Z)
    track2.rtk_pos[:, 2] = track2.rtk_pos[:, 2] - np.mean(track2.rtk_pos[:, 2])
    track2_comb = np.column_stack((track2.rtk_pos, track2.horz_acc, track2.vert_acc))

    # print(track2.X)
    # print(track2.Y)
    # print(track2.Z)
    track_combined = np.row_stack((track1_comb, track2_comb))
    # track_combined = np.transpose(
    #     np.hstack(
    #         (
    #             np.vstack(
    #                 (track1.X, track1.Y, track1.Z, track1.horz_acc, track1.vert_acc)
    #             ),
    #             np.vstack(
    #                 (track2.X, track2.Y, track2.Z, track2.horz_acc, track2.vert_acc)
    #             ),
    #         )
    #     )
    # )

    print(track_combined.shape)
    np.savetxt(
        util.DATA_DIR + "GPS RTK data/combined_coord_xsens.csv",
        track_combined,
        delimiter=",",
        header="X, Y, Z, horz_acc, vert_acc",
        comments="",
    )
    # # filename2 = util.DATA_DIR + "P3/EXCEL/easy_round.gnss"
    # # # print("STATUS \tloading track: {filename}".format(filename=filename))
    # # track2 = Track(filename2, "xsens")
    # # track2.load_track_data("gnss", coord_sys="xsens")
    # # print(track2.X[1:5])
    # # print(track2.Y[1:5])
    # # print(track2.Z)
    # # # track.train_elevation_model("xsens")
