import sys

sys.path.insert(0, "srcPython")
import datetime
import csv
import mvn  # Get constants
import util
import numpy as np
import quaternion as quat
from load_mvnx import load_mvnx
from segment import Segment
from track import Track

# from kalman import Kalman
from pose import Pose
from weather import Weather
import open3d as o3d  # To Do minimal import
import copy
import matplotlib.pyplot as plt
from matplotlib import use as plt_use
from statsmodels.nonparametric.smoothers_lowess import lowess
import random
from tqdm import tqdm

# from pose import Pose

plt_use("MacOSX")


class Subject:
    """
    Defines a subject and associated attributes
    """

    def __init__(self, subject_id):
        self.id = subject_id
        self.trial = None
        self.trial_date = None
        self.trial_time = None
        self.weather = None
        self.age = None
        self.gender = None
        self.weight = None
        self.height = None
        self.height_shoe = None
        self.shoe_length = None
        self.ski_length = None
        self.ski_contact_surface = None
        self.ski_rotation_point = None
        self.ski_width = None
        self.ski_type = None
        self.pole_length = None
        self.mvn = None
        self.mvnx = None
        self.gnss = None
        self.rtk = None
        self.image = None
        self.level = 0
        self.kalman = None
        self.segment_mdl = {}
        self.poses = []

        self.__load_subject()

    def __str__(self):
        return """Subject {self.id}:  
        \tGender = {self.gender} 
        \tAge = {self.age} 
        \tLevel = {self.level}
        \tWeight[kg] = {self.weight} 
        \tHeight[m] = {self.height}
        \t"Model = {self.segment_mdl}

        \tTrial: {self.trial}
        \t\tDate = {self.trial_date}
        \t\tTime = {self.trial_time} 

        \tEquipment: 
        \t\tSki type= {self.ski_type} 
        \t\tSki dimensions:
        \t\t\tLength[m] = {self.ski_length}
        \t\t\tWidth[m] = {self.ski_width}
        \t\t\tContact Surface[m] = {self.ski_contact_surface}
        \t\t\tRotation point[m] = {self.ski_rotation_point}

        \tData:
        \t\tMVN = {self.mvn}
        \t\tMVNX = {self.mvnx}
        \t\tGNSS = {self.gnss}
        \t\tRTK = {self.rtk}
        \t\tImage ={self.image}""".format(
            self=self
        )

    def __load_subject(self):
        with open(util.SUBJECT_TABLE, "r") as sfile:
            reader = csv.reader(sfile)
            header = next(reader)
            header[0] = util.strip_utf8(header[0])
            header[-1] = util.strip_endline(header[-1])

            found = False
            for r, row in enumerate(reader):
                if len(row) > 0:
                    found = True
                    if row[header.index("ParticipantID")] == str(self.id):
                        self.age = np.int16(row[header.index("Age")])
                        self.gender = np.int16(row[header.index("Gender")])
                        self.weight = np.float64(row[header.index("Weight")])
                        self.height = np.float64(row[header.index("Height")])
                        self.height_shoe = np.float64(row[header.index("HeightShoe")])
                        self.shoe_length = np.float64(row[header.index("ShoeLength")])
                        self.ski_length = np.float64(row[header.index("SkiLength")])
                        self.ski_contact_surface = np.float64(
                            row[header.index("SkiContactSurface")]
                        )
                        self.ski_rotation_point = np.float64(
                            row[header.index("SkiRotationPoint")]
                        )
                        self.ski_width = np.float64(row[header.index("SkiWidth")])
                        self.ski_type = row[header.index("EquipComments")]
                        self.pole_length = np.float64(row[header.index("PoleLength")])
                        self.level = np.int16(row[header.index("Level")])
                        date_time_string = row[header.index("DateTime")].strip()
                        date_time_obj = datetime.datetime.strptime(
                            date_time_string, "%Y-%m-%d %H:%M"
                        )
                        self.trial_date = date_time_obj.date()
                        self.trial_time = date_time_obj.time()

                        for s in util.SEGMENT_NAMES:
                            self.segment_mdl[s] = Segment(self, s)
            if not found:
                raise ValueError("{} not found in subject table.".format(self.id))

            self.weather = Weather(date_time_obj)

    def __load_data(self, dataset=None, **kwargs):
        """
        Loads a specific dataset
            :trial: <String> "easy", "medium", "hard"
            :dataset: <String> type of dataset "mvn", "mvnx", "gnss", "rtk", "image", "track"
        """

        # check arguments
        if self.trial not in ["easy", "medium", "hard"]:
            raise ValueError("{trial} type not valid".format(trial=self.trial))

        if dataset not in ["gnss", "rtk", "mvnx"]:
            raise ValueError("{dataset} not valid".format(dataset=dataset))

        if dataset == "mvn":
            print("WARNING: MVN dataset not currently processed")
            # TODO load mvn if necessary (Probably no
        elif dataset == "mvnx":
            print("STATUS: Loading mvnx data for subject {self.id}".format(self=self))
            file_name = (
                util.DATA_DIR
                + "P{self.id}/MVNX/".format(self=self)
                + self.trial
                + "_round.mvnx"
            )
            self.mvnx = load_mvnx(file_name)
        elif dataset == "gnss":
            print("STATUS: Loading gnss data for subject {self.id}".format(self=self))
            file_name = (
                util.DATA_DIR
                + "P{self.id}/EXCEL/".format(self=self)
                + self.trial
                + "_round.gnss"
            )
            self.gnss = Track(file_name, coord_sys=kwargs["coord_sys"])
            self.gnss.load_track_data(file_type="gnss", coord_sys=kwargs["coord_sys"])
            if kwargs["align_gns_rtk"]:
                print("STATUS: Aligning gnss with rtk")
                # align via ICP
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(
                    np.transpose(
                        np.vstack(
                            (
                                self.gnss.pos[:, 0] - np.mean(self.gnss.pos[:, 0]),
                                self.gnss.pos[:, 1] - np.mean(self.gnss.pos[:, 1]),
                                self.gnss.pos[:, 2],
                            )
                        )
                    )
                )
                source.paint_uniform_color([0.99, 0.49, 0.52])
                normals = np.zeros((len(self.gnss.pos[:, 0]), 3))
                normals[:, 2] = 1.0
                source.normals = o3d.utility.Vector3dVector(normals)

                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(
                    np.transpose(
                        np.vstack(
                            (
                                copy.deepcopy(self.rtk.rtk_pos[:, 0])
                                - np.mean(self.rtk.rtk_pos[:, 0]),
                                copy.deepcopy(self.rtk.rtk_pos[:, 1])
                                - np.mean(self.rtk.rtk_pos[:, 1]),
                                np.zeros((len(self.rtk.rtk_pos[:, 0]))),
                            )
                        )
                    )
                )
                target.paint_uniform_color([0.2, 0.2, 0.2])
                normals = np.zeros((len(self.rtk.rtk_pos[:, 0]), 3))
                normals[:, 2] = 1.0
                target.normals = o3d.utility.Vector3dVector(normals)

                init = np.identity(4)
                source_np = np.asarray(source.points)
                target_np = np.asarray(target.points)
                init[0:3, 3] = np.transpose(
                    np.mean(target_np, axis=0) - np.mean(source_np, axis=0)
                )

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source,
                    target,
                    50,
                    init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                source2 = copy.deepcopy(source)
                source2.transform(reg_p2p.transformation)
                source2 = np.asarray(source2.points)
                self.gnss.pos[:, 0] = source2[:, 0] + np.mean(self.gnss.pos[:, 0])
                self.gnss.pos[:, 1] = source2[:, 1] + np.mean(self.gnss.pos[:, 1])

        elif dataset == "rtk":
            if self.trial_date == datetime.date(2023, 2, 9):
                file_name = util.DATA_DIR + "GPS RTK data/P3_coord.csv"
            elif self.trial_date == datetime.date(2023, 2, 10):
                file_name = util.DATA_DIR + "GPS RTK data/P13_coord.csv"
            else:
                raise ValueError(
                    "No track data for trials on {self.trial_date}".format(self=self)
                )

            self.rtk = Track(file_name, coord_sys=kwargs["coord_sys"])
            self.rtk.load_track_data(file_type="rtk", coord_sys=kwargs["coord_sys"])
        elif dataset == "image":
            print(
                "WARNING: Image processing not currently supported.... Returning None"
            )

        else:
            raise ValueError("{dataset} type not valid.".format(dataset=dataset))

    # def __remove_drift(self, seg_vel, smooth_time=2.4):
    #     # Default smooth time is based on an average of 50 strides per second...
    #     # alternative = high pass filter at this cutoff freq...
    #     seg_vel = copy.deepcopy(seg_vel)
    #     frac = (self.mvnx.frame_rate * smooth_time) / self.mvnx.frame_count
    #     print("STATUS: Removing velocity drift.")

    #     # remove trend by lowess curve
    #     lws = np.zeros((self.mvnx.frame_count, 3))
    #     for i in range(3):
    #         lws[:, i] = np.asarray(lowess(seg_vel[:, i], self.gnss.time, frac=frac))[
    #             :, 1
    #         ]
    #         seg_vel[:, i] = seg_vel[:, i] - lws[:, i]
    #     # for j in range(3):
    #     #     seg_vel[:, j] = seg_vel[:, j] - np.mean(seg_vel[:, j])

    #     return seg_vel, lws

    def load_trial(self, trial_name, coord_sys="xsens", align_gns_rtk=True):
        if trial_name not in ["easy", "medium", "hard"]:
            raise ValueError("{self.trial} type not valid".format(self=self))

        self.trial = trial_name
        for dataset_type in ["rtk", "gnss", "mvnx"]:  # ["rtk", "gnss", "mvnx"]:  #
            self.__load_data(
                dataset_type, coord_sys=coord_sys, align_gns_rtk=align_gns_rtk
            )

        print("STATUS: Estimating pose")
        for frame in tqdm(range(self.mvnx.frame_count)):
            self.poses.append(self.get_pose(frame))
        # print("STATUS: setting up Kalman filter")
        # self.kalman = Kalman(self)

    def get_total_mass(self):
        return self.weight + 2 * self.get_ski_mass() + 2 * self.get_pole_mass()

    def get_ski_mass(self):
        return self.ski_length * self.ski_width * util.SKI_DENSITY

    def get_pole_mass(self):
        return self.pole_length * util.POLE_DENSITY

    def get_pose(self, frame):
        return Pose(self, frame)

    # def get_segment_acc(self, segment=-1, frame=mvn.FRAMES_ALL):
    #     if frame == mvn.FRAMES_ALL:
    #         seg_acc = np.zeros((self.mvnx.frame_count, 3))

    #         for i in range(self.mvnx.frame_count):
    #             if segment == -1:
    #                 seg_acc[i, :] = quat.rotate_vectors(
    #                     np.quaternion(
    #                         *self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, i)
    #                     ),
    #                     self.mvnx.get_center_of_mass_acc(i),
    #                 )
    #             else:
    #                 seg_acc[i, :] = quat.rotate_vectors(
    #                     np.quaternion(
    #                         *self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, i)
    #                     ),
    #                     self.mvnx.get_segment_acc(segment),
    #                 )
    #     else:
    #         seg_acc = np.zeros((3,))
    #         if segment == -1:
    #             seg_acc[i, :] = quat.rotate_vectors(
    #                 np.quaternion(
    #                     *self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, frame)
    #                 ),
    #                 self.mvnx.get_center_of_mass_acc(frame),
    #             )
    #         else:
    #             seg_acc[i, :] = quat.rotate_vectors(
    #                 quat.quaternion(
    #                     *self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, frame)
    #                 ),
    #                 self.mvnx.get_segment_acc(segment),
    #             )

    #     return seg_acc

    # def get_segment_vel(
    #     self, segment=-1, frame=mvn.FRAMES_ALL, remove_drift=True, **kwargs
    # ):
    #     seg_vel = np.zeros((self.mvnx.frame_count, 3))

    #     for i in range(self.mvnx.frame_count):
    #         if segment == -1:
    #             seg_vel[i, :] = quat.rotate_vectors(
    #                 np.quaternion(*self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, i)),
    #                 self.mvnx.get_center_of_mass_vel(i),
    #             )
    #         else:
    #             seg_vel[i, :] = quat.rotate_vectors(
    #                 np.quaternion(*self.mvnx.get_segment_ori(mvn.SEGMENT_PELVIS, i)),
    #                 self.mvnx.get_segment_vel(segment, i),
    #             )

    #     if remove_drift:
    #         if "smooth_time" in kwargs.keys():
    #             seg_vel, lws = self.__remove_drift(
    #                 seg_vel, smooth_time=kwargs["smooth_time"]
    #             )

    #         else:
    #             seg_vel, lws = self.__remove_drift(seg_vel)

    #         if frame != -1:
    #             return seg_vel[frame], lws[frame]
    #         else:
    #             return seg_vel, lws
    #     else:
    #         if frame != 1:
    #             return seg_vel[frame], None
    #         else:
    #             return seg_vel, None

    # def get_segment_pos(self, segment=-1, frame=mvn.FRAMES_ALL, **kwargs):
    #     dt = 1 / self.mvnx.frame_rate
    #     # TODO add 2pt rule to get center of mass position instead of the origin (current...)
    #     if segment == -1:
    #         print("STATUS: Computing CoM position in local frame")
    #     else:
    #         print(
    #             "STATUS: Computing {s} position.".format(
    #                 s=list(mvn.SEGMENTS.values())[segment]
    #             )
    #         )

    #     if frame == mvn.FRAMES_ALL:
    #         seg_pos = np.zeros((self.mvnx.frame_count + 1, 3))
    #         seg_vel, _ = self.get_segment_vel(segment, remove_drift=True)
    #         seg_pos[0, :] = self.mvnx.get_segment_pos(
    #             segment, 0
    #         )  # start at height within first obs
    #         for i in range(self.mvnx.frame_count):
    #             for j in range(3):
    #                 seg_pos[i + 1, j] = seg_pos[i, j] + (seg_vel[i, j] * dt)
    #         seg_pos2, lws = self.__remove_drift(seg_pos[1:, :])
    #         return seg_pos2, lws

    #     else:
    #         seg_pos = np.zeros((2, 3))
    #         seg_vel, _ = self.get_segment_vel(segment, frame, remove_drift=True)[
    #             i, :
    #         ]  # need to get all velocities to remove drift
    #         seg_pos[0, :] = self.mvnx.get_segment_pos(
    #             segment, 0
    #         )  # start at height in first obs
    #         for j in range(3):
    #             seg_pos[1, j] = seg_pos[0, j] + seg_vel * dt
    #         return seg_pos[1:, :]

    # def plot_pose(self, frame_idx, coord_sys="LCS", rotate=True):
    #     """
    #     Plot a subjects pose

    #         :mvnx_file: A mvnx file object
    #         :frame_idx: index of the desired frame to plot
    #         :return: A figure with the plot
    #     """

    #     if coord_sys not in ["LCS", "GCS"]:
    #         raise ValueError("{} not valid.".format(coord_sys))

    #     mvnx_file = self.mvnx

    #     # Get Data
    #     pelvis = mvnx_file.get_segment_pos(mvn.SEGMENT_PELVIS, frame_idx)
    #     l5 = mvnx_file.get_segment_pos(mvn.SEGMENT_L5, frame_idx)
    #     l3 = mvnx_file.get_segment_pos(mvn.SEGMENT_L3, frame_idx)
    #     t12 = mvnx_file.get_segment_pos(mvn.SEGMENT_T12, frame_idx)
    #     t8 = mvnx_file.get_segment_pos(mvn.SEGMENT_T8, frame_idx)
    #     neck = mvnx_file.get_segment_pos(mvn.SEGMENT_NECK, frame_idx)
    #     head = mvnx_file.get_segment_pos(mvn.SEGMENT_HEAD, frame_idx)

    #     l_shoulder = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_SHOULDER, frame_idx)
    #     l_upper_arm = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_UPPER_ARM, frame_idx)
    #     l_forearm = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_FOREARM, frame_idx)
    #     l_hand = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_HAND, frame_idx)

    #     l_upper_leg = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_UPPER_LEG, frame_idx)
    #     l_lower_leg = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_LOWER_LEG, frame_idx)
    #     l_foot = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_FOOT, frame_idx)
    #     l_toe = mvnx_file.get_segment_pos(mvn.SEGMENT_LEFT_TOE, frame_idx)

    #     r_shoulder = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_SHOULDER, frame_idx)
    #     r_upper_arm = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_UPPER_ARM, frame_idx)
    #     r_forearm = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_FOREARM, frame_idx)
    #     r_hand = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_HAND, frame_idx)

    #     r_upper_leg = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_UPPER_LEG, frame_idx)
    #     r_lower_leg = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_LOWER_LEG, frame_idx)
    #     r_foot = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_FOOT, frame_idx)
    #     r_toe = mvnx_file.get_segment_pos(mvn.SEGMENT_RIGHT_TOE, frame_idx)

    #     com = mvnx_file.get_center_of_mass_pos(frame_idx)
    #     com_vel = mvnx_file.get_center_of_mass_vel(frame_idx)
    #     com_acc = mvnx_file.get_center_of_mass_acc(frame_idx)

    #     ori = mvnx_file.get_segment_ori(mvn.SEGMENT_PELVIS, frame_idx)
    #     pelvis_q = np.quaternion(*ori)

    #     all_segments = [
    #         pelvis,
    #         l5,
    #         l3,
    #         t12,
    #         t8,
    #         neck,
    #         head,
    #         l_shoulder,
    #         r_shoulder,
    #         r_forearm,
    #         r_hand,
    #         r_upper_leg,
    #         r_lower_leg,
    #         r_foot,
    #         r_toe,
    #         l_forearm,
    #         l_hand,
    #         l_upper_leg,
    #         l_lower_leg,
    #         l_foot,
    #         l_toe,
    #     ]

    #     if rotate:
    #         for i, seg in enumerate(all_segments):
    #             all_segments[i] = quat.rotate_vectors(1 / pelvis_q, seg)

    #     # if coord_sys == "GCS":
    #     #     LCS_pos, LCS_vel, LCS_acc = self.kalman.predict(self, 0, frame_idx)
    #     #     # com += LCS_pos
    #     #     com_vel += LCS_vel
    #     #     com_acc += LCS_acc
    #     # print(ori)
    #     com_vel = quat.rotate_vectors(pelvis_q, com_vel)
    #     com_acc = quat.rotate_vectors(pelvis_q, com_acc)
    #     # print("blah")

    #     # if coord_sys == "GCS":
    #     #     com_acc = com_acc / np.abs(util.GRAVITY)
    #     # elif coord_sys == "LCS":
    #     # com += mvnx_file.get_segment_pos(0, frame_idx)

    #     blue_segments = all_segments[0:8]
    #     right_segments = all_segments[8:13]
    #     left_segments = all_segments[13:]

    #     # Set up plot
    #     fig = plt.figure(0)
    #     ax = plt.axes(projection="3d")
    #     ax.set_xlabel("X[m]")
    #     ax.set_ylabel("Y[m]")
    #     ax.set_zlabel("Z[m]")
    #     ax.set_title(
    #         "P{self.id} - {self.trial}: Pose in {cs} at frame {frame} s".format(
    #             self=self, cs=coord_sys, frame=frame_idx
    #         )
    #     )
    #     # scale factor for vectors
    #     scale = 10
    #     # Plot points (left = red, right = green)
    #     for seg in blue_segments:

    #         ax.scatter(seg[0], seg[1], seg[2], color=util.CMAP(0))

    #     for seg in right_segments:
    #         ax.scatter(seg[0], seg[1], seg[2], color=util.CMAP(2))

    #     for seg in left_segments:
    #         ax.scatter(seg[0], seg[1], seg[2], color=util.CMAP(3))

    #     ax.scatter(
    #         com[0],
    #         com[1],
    #         com[2],
    #         s=50,
    #         color="black",
    #     )
    #     vel_arrow_prop_dict = dict(
    #         mutation_scale=3, arrowstyle="-|>", color="green"  # , shrinkA=0, shrinkB=0
    #     )

    #     acc_arrow_prop_dict = dict(
    #         mutation_scale=3, arrowstyle="-|>", color="red"  # , shrinkA=0, shrinkB=0
    #     )
    #     print(
    #         "COM Position: [{com_pos[0]}, {com_pos[1]}, {com_pos[2]}] m".format(
    #             com_pos=com
    #         )
    #     )
    #     print(
    #         "COM Velocity: [{com_vel[0]}, {com_vel[1]}, {com_vel[2]}] m/s".format(
    #             com_vel=com_vel
    #         )
    #     )
    #     print(
    #         "COM Acceleration: [{com_acc[0]}, {com_acc[1]}, {com_acc[2]}] m/s/s".format(
    #             com_acc=com_acc
    #         )
    #     )
    #     # TODO fix vel/acceleration arrow display rotations messing with direction...
    #     com_vel_arrow = util.Arrow3D(
    #         [com[0], scale * com_vel[0]],
    #         [com[1], scale * com_vel[1]],
    #         [com[2], scale * com_vel[2]],
    #         **vel_arrow_prop_dict,
    #     )

    #     com_acc_arrow = util.Arrow3D(
    #         [com[0], scale * com_acc[0]],
    #         [com[1], scale * com_acc[1]],
    #         [com[2], scale * com_acc[2]],
    #         **acc_arrow_prop_dict,
    #     )
    #     ax.add_artist(com_vel_arrow)
    #     ax.add_artist(com_acc_arrow)

    #     # Fix aspect ratio
    #     limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    #     ax.set_box_aspect(np.ptp(limits, axis=1))
    #     plt.show(block=False)  # show plot
    #     return fig, ax


if __name__ == "__main__":
    print("Main subject...")
    sub = Subject(3)
    sub.load_trial("easy", "xsens")
    print(sub)

    # frame =29000
    # # Determine pelvis orientation to rotate to plane in direction of travel (approximatly...)
    # pelvis = sub.mvnx.get_segment_pos(mvn.SEGMENT_PELVIS, frame)
    # l_hip = sub.mvnx.get_segment_pos(mvn.SEGMENT_LEFT_UPPER_LEG, frame)
    # r_hip = sub.mvnx.get_segment_pos(mvn.SEGMENT_RIGHT_UPPER_LEG, frame)

    # v1 = r_hip - pelvis
    # v3 = np.array([0, 0, 1])
    # v2 = np.cross(v3, v1)
    # v1 = np.cross(v2,v3)
    # v1 = v1/np.linalg.norm(v1)
    # v2 = v2/np.linalg.norm(v2)
    # v3 = v3/np.linalg.norm(v3)

    # ang = np.arccos(v2[0]/ np.linalg.norm(v2))
    # q = quat.from_euler_angles([0,0,ang])

    # # Project to frontal plane
    # points = []
    # for i, s in enumerate(mvn.SEGMENTS.keys()):
    #     pt = quat.rotate_vectors(1/q, sub.mvnx.get_segment_pos(s, frame))
    #     points.append([0,  pt[1], pt[2]])
    #     points.append([1,  pt[1], pt[2]])
    # points = np.array(points)

    # # CoSet as  point cloud
    # pt_cloud = o3d.geometry.PointCloud()
    # pt_cloud.points = o3d.utility.Vector3dVector(
    #                 points
    #             )
    # hull, a = pt_cloud.compute_convex_hull()
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    # hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([pt_cloud, hull_ls])
    # area = hull.get_volume()

    # ax = plt.subplot()
    # for i in range(len(points)):
    #     ax.scatter(points[i,1], points[i,2])
    # plt.show()

    # ax = plt.axes(projection="3d")
    # for i in range(trunk_key_points.shape[0]):
    #     r = trunk_key_points[i,:]
    #     r2 = quat.rotate_vectors(1/q, trunk_key_points[i,:])
    #     if i ==0:
    #         ax.scatter(r[0], r[1], r[2], color="black")
    #     else:
    #         ax.scatter(r[0], r[1], r[2], color=util.CMAP(0))
    #     ax.scatter(r2[0], r2[1], r2[2], color="gray")

    # ax.scatter(r_hip[0], r_hip[1], r_hip[2], color="green")
    # ax.scatter(l_hip[0], l_hip[1], l_hip[2], color="red")

    # p1 = util.Arrow3D(
    #         [pelvis[0], pelvis[0]+v1[0]],
    #         [pelvis[1], pelvis[1]+v1[1]],
    #         [pelvis[2], pelvis[2]+v1[2]],
    #     **dict(mutation_scale=3,  arrowstyle="-|>", color="red")
    # )
    # v1_prime = quat.rotate_vectors(1/q, v1)
    # p1_prime = util.Arrow3D(
    #         [pelvis[0], pelvis[0]+v1_prime[0]],
    #         [pelvis[1], pelvis[1]+v1_prime[1]],
    #         [pelvis[2], pelvis[2]+v1_prime[2]],
    #     **dict(mutation_scale=3,  arrowstyle="-|>", color="red")
    # )
    # p2 = util.Arrow3D(
    #         [pelvis[0], pelvis[0]+v2[0]],
    #         [pelvis[1], pelvis[1]+v2[1]],
    #         [pelvis[2], pelvis[2]+v2[2]],
    #     **dict(mutation_scale=3,  arrowstyle="-|>", color="green")
    # )
    # p3 = util.Arrow3D(
    #         [pelvis[0], pelvis[0]+v3[0]],
    #         [pelvis[1], pelvis[1]+v3[1]],
    #         [pelvis[2], pelvis[2]+v3[2]],
    #     **dict(mutation_scale=3,  arrowstyle="-|>", color="blue")
    # )
    # ax.add_artist(p1)
    # ax.add_artist(p2)
    # ax.add_artist(p3)
    # ax.add_artist(p1_prime)
    # ax.set_xlabel("X[m]")
    # ax.set_ylabel("Y[m]")
    # ax.set_zlabel("Z[m]")
    # limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    # ax.set_box_aspect(np.ptp(limits, axis=1))
    # plt.show(block=True)

    # ####### END PLOT

    # pt_cloud =
    # r_shoulder = sub.mvnx.get_segment_pos(mvn.SEGMENT_RIGHT_SHOULDER, frame)
    # _shoulder = sub.mvnx.get_segment_pos(mvn.SEGMENT_RIGHT_SHOULDER, frame)
    # r_shoulder = sub.mvnx.get_segment_pos(mvn.SEGMENT_RIGHT_SHOULDER)

    # # frame = 1500
    # # coord_sys = "LCS"
    # # fig = sub.plot_pose(frame, coord_sys)
    # # print("STATUS: Saving pose plot to output directory.")
    # # plt.savefig(
    # #     util.DATA_DIR
    # #     + "output/P{sub.id}_{sub.trial}_frame_{f}_pose_{coord}.png".format(
    # #         sub=sub, f=frame, coord=coord_sys
    # #     )
    # # )

    # # coord_sys = "GCS"
    # # fig = sub.plot_pose(frame, coord_sys)
    # # print("STATUS: Saving pose plot to output directory.")
    # # plt.savefig(
    # #     util.DATA_DIR
    # #     + "output/P{sub.id}_{sub.trial}_frame_{f}_pose_{coord}.png".format(
    # #         sub=sub, f=frame, coord=coord_sys
    # #     )
    # # )

    # # com = np.asarray(sub.mvnx.get_center_of_mass_pos())
    # # print(com)
    # # # fig = plt.figure(0)
    # # ax = fig.add_subplot(projection="3d")
    # # # ax = plt.axes()
    # # # ax.plot(sub.rtk.X, sub.rtk.Y, color=util.CMAP(0))
    # # ax.scatter(sub.gnss.X, sub.gnss.Y, sub.gnss.Z, color=util.CMAP(0))
    # # # ax.scatter(
    # # #     sub.rtk.X,
    # # #     sub.rtk.Y,
    # # #     sub.rtk.Z - np.mean(sub.rtk.Z),
    # # #     color=util.CMAP(1),
    # # # )
    # # plt.show()
    # # print(sub)

    # # file_name = util.DATA_DIR + "P3/MVNX/" + "easy" + "_round.mvnx"
    # # print("LOADING MVNX FOR: {filename}".format(filename=file_name))
    # # mvnx = load_mvnx(file_name)
    # # a_s = mvnx.get_sensor_free_acc(0, 0)
    # # q_s = mvnx.get_sensor_ori(0, 0)

    # # util.rotate_quat(q_s, a_s)
    # # a_g = q_s*a_s*

    # # print(mvnx)
