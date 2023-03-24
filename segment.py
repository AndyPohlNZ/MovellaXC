# Rigid link model for a subject.


""" 
Compute a rigid link model for a given subject given DeLava's equations

# 15 Links:
Foot + ski
L/R shank
L/R thigh
L/R hand + pole
L/R forearm
L/R upperarm
Head + neck
Trunk upper
Trunk Mid
Trunk lower.
"""

import sys

sys.path.insert(0, "srcPython")
import util
import re


class Segment:
    """Defines a segment in a rigid body model"""

    def __init__(self, sub, name):

        if name not in util.SEGMENT_NAMES:
            raise ValueError("Segment {} is not defined".format(name))

        self.name = name
        self.subject = sub
        self.mass = None
        self.length = None
        self.com = None
        self.inertia = None

        self.__get_segment_mass(name)
        self.__get_segment_length(name)
        self.__get_segment_com(name)
        self.__get_inertia(name)

    def __str__(self):
        return """Segment: {self.name}:\n\t
        Mass: {self.mass}kg\n\t
        Length: {self.length}m\n\t
        COM: {self.com}m\n\t
        Moment of Inertia: {self.inertia}kgm^2
        """.format(
            self=self
        )

    def __get_segment_mass(self, name):
        name = self.__strip_side(name)

        if self.subject.gender == 0:  # male
            self.mass = self.subject.weight * util.SEGMENT_MASS_MALE[name]
        else:
            self.mass = self.subject.weight * util.SEGMENT_MASS_FEMALE[name]

        if name == "foot":
            self.mass += self.__get_ski_mass()

        if name == "hand":
            self.mass += self.__get_pole_mass()

    def __get_segment_length(self, name):
        # TODO get these from t pose rather than as a proportion of average human
        name = self.__strip_side(name)
        if self.subject.gender == 0:  # male
            self.length = self.subject.height * util.SEGMENT_LENGTH_MALE[name]
        else:
            self.length = self.subject.height * util.SEGMENT_LENGTH_FEMALE[name]

    def __get_segment_com(self, name):
        name = self.__strip_side(name)
        if self.subject.gender == 0:  # male
            self.com = self.length * util.SEGMENT_COM_MALE[name]
        else:
            self.com = self.length * util.SEGMENT_COM_FEMALE[name]

    def __get_inertia(self, name):
        name = self.__strip_side(name)
        if self.subject.gender == 0:  # male
            self.inertia = (
                self.mass * (self.length * util.SEGMENT_RADII_GYRATION_MALE[name]) ** 2
            )
        else:
            self.inertia = (
                self.mass
                * (self.length * util.SEGMENT_RADII_GYRATION_FEMALE[name]) ** 2
            )

    def __strip_side(self, name):
        name = re.sub("^(l_)", "", name)
        name = re.sub("^(r_)", "", name)
        return name

    def __get_pole_mass(self):
        return self.subject.pole_length * util.POLE_DENSITY

    def __get_ski_mass(self):
        return self.subject.ski_length * self.subject.ski_width * util.SKI_DENSITY


# testing

if __name__ == "__main__":
    from subject import Subject

    segment = "r_upper_leg"
    print("STATUS: Generating segment for {}".format(segment))
    sub = Subject(3)
    segment = Segment(sub, segment)
    print(segment)
