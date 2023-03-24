import os, sys

sys.path.insert(0, "srcPython")
import mvn
import util
import numpy as np
import quaternion as quat
import open3d as o3d  # To Do minimal import
import matplotlib.pyplot as plt
import copy


class Pose:
    """Defines a subjects pose at a given frame:
    :sub: <Subject> subject.
    :frame: <int> frame number of trial to compute.
    """

    def __init__(self, sub, frame):

        self.frame = frame
        self.points = {}
        self.vectors = {}
        self.area = None
        self.volume = None
        self.coordinate_system = "Xsens"  # Xsens, GCS, LCS
        self.__get_points(sub)
        self.__get_xsens_ori()
        self.__to_ground()
        self.__to_lcs()

        self.__get_volume()
        # self.__get_gcs_kin(sub)
        # self.__to_gcs()
        try:
            self.__get_area()
        except:
            print("WARNING: An error occured in estimating area skipping calculation")

    def __str__(self):
        com = self.get_com()
        vel = self.get_com_vel()
        acc = self.get_com_acc()
        origin = self.__get_origin()
        ori = self.__get_ori()

        return """
        Pose at frame {self.frame}:
        \t Coordinate System: {self.coordinate_system}
        \t Area: {self.area} m^2
        \t Volume: {self.volume} m^3
        \t CoM: {com} m
        \t Com Vel: {vel} m/s
        \t Com Acc: {acc} m/s/s
        \t Global Origin: {origin} m
        \t Direction: {ori}
        """.format(
            self=self,
            com=self.get_com(),
            vel=self.get_com_vel(),
            acc=self.get_com_acc(),
            origin=self.__get_point("Origin"),
            ori=self.__get_vector("Ori"),
        )

    def __get_points(self, sub):
        for seg in mvn.SEGMENTS.items():
            if seg[1] == "com":
                # Get CoM accounting for mass of skis/poles
                self.points["CoM"] = (
                    sub.mvnx.get_segment_pos(seg[0], self.frame)
                    # + (
                    #     sub.mvnx.get_segment_pos(mvn.SEGMENT_LEFT_HAND, self.frame)
                    #     / sub.get_pole_mass()
                    # )
                    # + (
                    #     sub.mvnx.get_segment_pos(mvn.SEGMENT_RIGHT_HAND, self.frame)
                    #     / sub.get_pole_mass()
                    # )
                    # + (
                    #     sub.mvnx.get_segment_pos(mvn.SEGMENT_LEFT_TOE, self.frame)
                    #     / sub.get_ski_mass()
                    # )
                    # + (
                    #     / sub.get_ski_mass()
                    # )
                )
            else:
                self.points[seg[1]] = sub.mvnx.get_segment_pos(seg[0], self.frame)
        self.points["Origin"] = sub.mvnx.get_segment_pos(
            mvn.SEGMENT_CENTER_OF_MASS, self.frame
        )

        self.vectors["XsensCoMVel"] = sub.mvnx.get_center_of_mass_vel(self.frame)
        self.vectors["XsensCoMAcc"] = sub.mvnx.get_center_of_mass_acc(self.frame)
        self.vectors["XsensOri"] = np.quaternion(
            *sub.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, self.frame)
        )

    def __get_point(self, point):
        return copy.deepcopy(self.points[point])

    def __get_vector(self, vector):
        return copy.deepcopy(self.vectors[vector])

    def get_com(self):
        return copy.deepcopy(self.points["CoM"])

    def __get_origin(self):
        return copy.deepcopy(self.points["Origin"])

    def get_com_vel(self):
        com_vel = None
        if self.coordinate_system == "GCS":
            com_vel = copy.deepcopy(self.vectors["XsensCoMVel"]) + copy.deepcopy(
                self.vectors["GCSVel"]
            )
        elif self.coordinate_system == "LCS":
            com_vel = copy.deepcopy(self.vectors["XsensCoMVel"])
        return com_vel

    def get_com_acc(self):
        com_acc = None
        if self.coordinate_system == "GCS":
            com_acc = copy.deepcopy(self.vectors["XsensCoMAcc"]) + copy.deepcopy(
                self.vectors["GCSAcc"]
            )
        elif self.coordinate_system == "LCS":
            com_acc = copy.deepcopy(self.vectors["XsensCoMAcc"])
        return com_acc + [0, 0, -util.GRAVITY]

    def __get_ori(self):
        return copy.deepcopy(self.vectors["Ori"])

    def __to_lcs(self):
        # for i, k in enumerate(self.points.keys()):
        #     self.points[k] = self.points[k] - self.get_com()
        #
        for i, k in enumerate(self.points.keys()):
            self.points[k] = quat.rotate_vectors(
                self.vectors["LCSOri"], self.__get_point(k)
            )
        self.coordinate_system = "LCS"

    def __to_gcs(self):
        if self.coordinate_system == "GCS":
            print("WARNING: pose already in GCS - doing nothing")

        elif self.coordinate_system != "LCS":
            self.__to_lcs()
            ori = self.vectors["Ori"]
            for i, k in enumerate(self.points.keys()):
                if k != "GCSOrigin":
                    self.points[k] = (
                        quat.rotate_vectors(self.vectors["Ori"], self.__get_point(k))
                        + self.points["GCSOrigin"]
                    )
            for i, k in enumerate(self.vectors.keys()):
                if k in ["XsensCoMVel", "XsensCoMAcc"]:
                    self.vectors[k] = quat.rotate_vectors(ori, self.__get_vector(k))
        else:
            ori = self.vectors["Ori"]
            for i, k in enumerate(self.points.keys()):
                if k != "GCSOrigin":
                    self.points[k] = (
                        quat.rotate_vectors(ori, self.__get_point(k))
                        + self.points["GCSOrigin"]
                    )

            for i, k in enumerate(self.vectors.keys()):
                if k in ["XsensCoMVel", "XsensCoMAcc"]:
                    self.vectors[k] = quat.rotate_vectors(ori, self.__get_vector(k))

        self.coordinate_system = "GCS"

    def __to_ground(self):
        min_height = 1e12
        # pts = copy.deepcopy(self.points)
        for i, k in enumerate(self.points.keys()):
            if k not in ["Origin", "CoM"]:
                if self.points[k][2] < min_height:
                    min_height = self.points[k][2]

        if min_height != 0:
            for i, k in enumerate(self.points.keys()):
                if k == "CoM":
                    com = self.get_com()
                    com[2] = com[2] - min_height
                    self.points["CoM"] = copy.deepcopy(com)
                elif k != "CoM":
                    self.points[k][2] = self.points[k][2] - min_height

    def plot_area(self):
        point_vol = []
        for i, k in enumerate(self.points.keys()):
            pt = self.__get_point(k)
            point_vol.append([pt[0], 0, pt[2]])
            point_vol.append([pt[0], 1, pt[2]])
        point_vol = np.array(point_vol)

        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(point_vol)
        tmp_norms = np.zeros((point_vol.shape[0], 3))
        tmp_norms[:, 1] = 1.0
        pt_cloud.normals = o3d.utility.Vector3dVector(tmp_norms)
        hull, _ = pt_cloud.compute_convex_hull()
        # convert to point cloud
        ax = plt.subplot()
        for i in range(point_vol.shape[0]):
            ax.scatter(point_vol[i, 0], point_vol[i, 2], color="grey")
        ax.scatter(point_vol[15, 0], point_vol[15, 2], color="green")
        ax.scatter(point_vol[19, 0], point_vol[19, 2], color="red")

        ax.set_aspect("equal")
        ax.set_title("Frontal Plane area ")
        plt.show()

        return ax, hull

    def __get_area(self, debug=False):
        # Generate point cloud.  Note projection by 1unit in x direction
        # so that area = volume
        point_vol = []
        for i, k in enumerate(self.points.keys()):
            pt = self.__get_point(k)
            point_vol.append([pt[0], 0, pt[2]])
            point_vol.append([pt[0], 1, pt[2]])
        point_vol = np.array(point_vol)

        # convert to point cloud
        if debug:
            ax = plt.subplot()
            for i in range(point_vol.shape[0]):
                ax.scatter(point_vol[i, 0], point_vol[i, 2])
            ax.scatter(point_vol[15, 0], point_vol[15, 2], color="green")
            ax.scatter(point_vol[19, 0], point_vol[19, 2], color="red")

            ax.set_aspect("equal")
            ax.set_title("Frontal Plane area calc")
            plt.show()

        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(point_vol)
        tmp_norms = np.zeros((point_vol.shape[0], 3))
        tmp_norms[:, 1] = 1.0
        pt_cloud.normals = o3d.utility.Vector3dVector(tmp_norms)
        hull, _ = pt_cloud.compute_convex_hull()

        self.area = hull.get_volume() / 1

    def __get_volume(self):
        pts = []
        for i, k in enumerate(self.points.keys()):
            pts.append(self.__get_point(k))

        pts = np.asarray(pts)
        # create convex hull
        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(pts)
        hull, _ = pt_cloud.compute_convex_hull()

        self.volume = hull.get_volume()

    def __parse_coordinate_systems(self, cs):
        if cs not in ["GCS", "LCS", "Xsens"]:
            raise ValueError(
                '{} is not a valid coordinate system.  \nValid coordinate systems are ["GCS", "LCS", "Xsens"]'
            )

    def __get_xsens_ori(self):
        pelvis = self.points["Pelvis"]
        r_hip = self.points["RightUpperLeg"]

        # rotate and project given orientation of pelvis.
        v1 = r_hip - pelvis
        v3 = np.array([0, 0, 1])
        v2 = np.cross(v3, v1)
        v1 = np.cross(v2, v3)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)

        self.vectors["LCSOri"] = util.rel_quat(
            v1, [1, 0, 0]
        )  # make v1 the new x axis...

    def __get_gcs_kin(self, sub):
        ori = sub.gnss.vel[self.frame, :]

        self.vectors["Ori"] = util.rel_quat(ori, [1, 0, 0])
        self.points["GCSOrigin"] = sub.gnss.pos[self.frame, :]

        self.vectors["GCSVel"] = sub.gnss.vel[self.frame, :]
        self.vectors["GCSAcc"] = sub.gnss.acc[self.frame, :]

    def plot_pose(self, title=""):
        """
        Plot a subjects pose
        """

        if self.coordinate_system == "GCS":
            xlabel = "X"
            ylabel = "Y"
        else:
            xlabel = "X [m]"
            ylabel = "Y [m]"

        zlabel = "Up [m]"

        pts = self.points

        fig = plt.figure(0)
        ax = plt.axes(projection="3d")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, k in enumerate(pts.keys()):
            pt = pts[k]
            if k == "RightShoulder":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))
            elif k == "LeftShoulder":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))
            elif k[:4] == "Righ":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(2))
            elif k[:4] == "Left":
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(3))
            elif k[:4] == "CoM":
                ax.scatter(pt[1], pt[0], pt[2], color="black")
            elif k[:4] == "Orig":
                break
            else:
                ax.scatter(pt[1], pt[0], pt[2], color=util.CMAP(0))

            print("Point{p}: {pp}".format(p=k, pp=pt))

        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["L5"][1]],
                [pts["Pelvis"][0], pts["L5"][0]],
                [pts["Pelvis"][2], pts["L5"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["L5"][1], pts["L3"][1]],
                [pts["L5"][0], pts["L3"][0]],
                [pts["L5"][2], pts["L3"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["L3"][1], pts["T12"][1]],
                [pts["L3"][0], pts["T12"][0]],
                [pts["L3"][2], pts["T12"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T12"][1], pts["Neck"][1]],
                [pts["T12"][0], pts["Neck"][0]],
                [pts["T12"][2], pts["Neck"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["Head"][1]],
                [pts["Neck"][0], pts["Head"][0]],
                [pts["Neck"][2], pts["Head"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T8"][1], pts["RightShoulder"][1]],
                [pts["T8"][0], pts["RightShoulder"][0]],
                [pts["T8"][2], pts["RightShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["T8"][1], pts["LeftShoulder"][1]],
                [pts["T8"][0], pts["LeftShoulder"][0]],
                [pts["T8"][2], pts["LeftShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["RightShoulder"][1]],
                [pts["Neck"][0], pts["RightShoulder"][0]],
                [pts["Neck"][2], pts["RightShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["Neck"][1], pts["LeftShoulder"][1]],
                [pts["Neck"][0], pts["LeftShoulder"][0]],
                [pts["Neck"][2], pts["LeftShoulder"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(0)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["LeftShoulder"][1], pts["LeftUpperArm"][1]],
                [pts["LeftShoulder"][0], pts["LeftUpperArm"][0]],
                [pts["LeftShoulder"][2], pts["LeftUpperArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftUpperArm"][1], pts["LeftForeArm"][1]],
                [pts["LeftUpperArm"][0], pts["LeftForeArm"][0]],
                [pts["LeftUpperArm"][2], pts["LeftForeArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftForeArm"][1], pts["LeftHand"][1]],
                [pts["LeftForeArm"][0], pts["LeftHand"][0]],
                [pts["LeftForeArm"][2], pts["LeftHand"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["RightShoulder"][1], pts["RightUpperArm"][1]],
                [pts["RightShoulder"][0], pts["RightUpperArm"][0]],
                [pts["RightShoulder"][2], pts["RightUpperArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightUpperArm"][1], pts["RightForeArm"][1]],
                [pts["RightUpperArm"][0], pts["RightForeArm"][0]],
                [pts["RightUpperArm"][2], pts["RightForeArm"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightForeArm"][1], pts["RightHand"][1]],
                [pts["RightForeArm"][0], pts["RightHand"][0]],
                [pts["RightForeArm"][2], pts["RightHand"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["LeftUpperLeg"][1]],
                [pts["Pelvis"][0], pts["LeftUpperLeg"][0]],
                [pts["Pelvis"][2], pts["LeftUpperLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftUpperLeg"][1], pts["LeftLowerLeg"][1]],
                [pts["LeftUpperLeg"][0], pts["LeftLowerLeg"][0]],
                [pts["LeftUpperLeg"][2], pts["LeftLowerLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftLowerLeg"][1], pts["LeftFoot"][1]],
                [pts["LeftLowerLeg"][0], pts["LeftFoot"][0]],
                [pts["LeftLowerLeg"][2], pts["LeftFoot"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["LeftFoot"][1], pts["LeftToe"][1]],
                [pts["LeftFoot"][0], pts["LeftToe"][0]],
                [pts["LeftFoot"][2], pts["LeftToe"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(3)),
            )
        )

        ax.add_artist(
            util.Arrow3D(
                [pts["Pelvis"][1], pts["RightUpperLeg"][1]],
                [pts["Pelvis"][0], pts["RightUpperLeg"][0]],
                [pts["Pelvis"][2], pts["RightUpperLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightUpperLeg"][1], pts["RightLowerLeg"][1]],
                [pts["RightUpperLeg"][0], pts["RightLowerLeg"][0]],
                [pts["RightUpperLeg"][2], pts["RightLowerLeg"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightLowerLeg"][1], pts["RightFoot"][1]],
                [pts["RightLowerLeg"][0], pts["RightFoot"][0]],
                [pts["RightLowerLeg"][2], pts["RightFoot"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        ax.add_artist(
            util.Arrow3D(
                [pts["RightFoot"][1], pts["RightToe"][1]],
                [pts["RightFoot"][0], pts["RightToe"][0]],
                [pts["RightFoot"][2], pts["RightToe"][2]],
                **dict(mutation_scale=3, arrowstyle="-", color=util.CMAP(2)),
            )
        )
        # com_vel = self.get_com_vel()
        # com_acc = self.get_com_acc()
        # com = self.get_com()
        # if com_vel is not None:
        #     print("COM = {}".format(com))
        #     print("CoM Velocity = {}".format(com_vel))
        #     ax.add_artist(
        #         util.Arrow3D(
        #             [com[1], com[1] + com_vel[1]],
        #             [com[0], com[0] + com_vel[0]],
        #             [com[2], com[2] + com_vel[2]],
        #             **dict(mutation_scale=3, arrowstyle="-|>", color="green"),
        #         )
        #     )
        # if com_acc is not None:
        #     print("CoM Acceleration = {}".format(com_acc))
        #     ax.add_artist(
        #         util.Arrow3D(
        #             [com[1], com[1] + com_acc[1]],
        #             [com[0], com[0] + com_acc[0]],
        #             [com[2], com[2] + com_acc[2]],
        #             **dict(mutation_scale=3, arrowstyle="-|>", color="red"),
        #         )
        #     )

        limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
        ax.set_box_aspect(np.ptp(limits, axis=1))
        ax.set_title(title)
        plt.show(block=True)  # show plot
        return fig, ax


if __name__ == "__main__":
    from subject import Subject

    sub = Subject(3)
    sub.load_trial("hard", "xsens")

    frame = 32301
    frame = 43500
    p = Pose(sub, frame)

    vel = p.get_com_vel()
    com = p.get_com()
    ax = plt.subplot()
    ax.plot(sub.gnss.pos[:, 1], sub.gnss.pos[:, 0])
    ax.scatter(sub.gnss.pos[frame, 1], sub.gnss.pos[frame, 0])
    ax.scatter(com[1], com[0], color="green")
    ax.arrow(com[1], com[0], 5 * vel[1], 5 * vel[0], color="green")
    plt.show()

    p = Pose(sub, 43550)
    p = Pose(sub, 43600)
    p = Pose(sub, 43650)

    p2 = Pose(sub, 28000)

    frames = list(range(0, sub.mvnx.frame_count, 2000))
    poses = []
    for f in frames:
        poses.append(Pose(sub, f))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(sub.gnss.pos[:, 0], sub.gnss.pos[:, 1])
    scale = 1
    for p in poses:
        com = p.get_com()
        com_vel = p.get_com_vel()
        com_acc = p.get_com_acc() / util.GRAVITY  # convert to gs
        print(p.get_com())
        print(p.get_com_vel())
        ax1.arrow(com[0], com[1], scale * com_vel[0], scale * com_vel[1], color="green")
        ax1.arrow(com[0], com[1], com_acc[0], com_acc[1], color="red")

        ax2.arrow(com[1], com[2], scale * com_vel[1], scale * com_vel[2], color="green")
        ax2.arrow(com[1], com[2], com_acc[1], com_acc[2], color="red")
    fig.show()

    p = Pose(sub, 0)
    p.plot_pose()
