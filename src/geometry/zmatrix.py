import numpy
import typing
from src.geometry import base


class ZMatrix(base.BaseGeometry):
    def __init__(self):
        super().__init__()

    @property
    def geometry(self) -> dict:
        """
        Get the molecular geometry in Cartesian representation.

        Returns:
            tuple:
                A tuple containing:
                - atoms (list[str]): Atomic symbols.
                - coords (numpy.ndarray): Cartesian coordinates (N, 3).
        """
        return {
            "atoms": self.atoms, 
            "coords": self.coords,
            "charge": self.charge,
            "multi": self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses": self.atomicmass
        }

    @geometry.setter
    def geometry(self, value) -> typing.Any:
        _coords = []
        if isinstance(value, str):
            _structure = [_line for _line in value.split('\n') if _line.strip()]
            print(_structure)
            self.natoms = int(_structure[0].split()[0])
            self.charge, self.multi = map(int, _structure[1].split())

            if len(_structure[2:]) < 1:
                raise ValueError("No atoms have been defined. Please define atleast one atom in the input block.")

            # now process the first atom
            self.atoms.append(_structure[2].split()[0])
            _coords.append([0, 0, 0])

            # now process second atom
            if len(_structure[3:]) >= 1:
                self.atoms.append(_structure[3].split()[0])
                # second atom is always placed on x-axis and will be connected to first
                _distance = float(_structure[3].split()[-1])
                _coords.append([_distance, 0, 0])

            # now process third atom (if present)
            if len(_structure[4:]) >= 1:
                self.atoms.append(_structure[4].split()[0])

                # third atom is always placed on xy plane and will be connected to first or second atom
                _distance = float(_structure[4].split()[2])
                _angle = float(_structure[4].split()[4])

                _bondedTo = int(_structure[4].split()[1]) - 1   # can be 0 (first atom) or 1 (second atom)
                _angleFrom = int(_structure[4].split()[3]) - 1

                # compute bond vector relative to chosen bonded atom
                _bondVector = numpy.array(_coords[_bondedTo]) - numpy.array(_coords[_angleFrom])

                # normalize bond vector
                _bondVector = _bondVector / numpy.linalg.norm(_bondVector)

                # construct a perpendicular vector in xy-plane
                _perp = numpy.array([_bondVector[1], -_bondVector[0], 0.0])
                _perp = _perp / numpy.linalg.norm(_perp)

                # rotate bond vector by angle
                _rotated = (numpy.cos(numpy.radians(_angle)) * _bondVector + numpy.sin(numpy.radians(_angle)) * _perp)

                # scale by distance and place new atom
                _newCoord = numpy.array(_coords[_bondedTo]) + _distance * _rotated
                _coords.append(_newCoord.tolist())

            # now process any other atoms (if present)
            for i in range(5, len(_structure)):
                tokens = _structure[i].split()
                self.atoms.append(tokens[0])

                _distance = float(tokens[2])
                _angle = float(tokens[4])
                _dihedral = float(tokens[6])

                _bondedTo = int(tokens[1]) - 1
                _angleFrom = int(tokens[3]) - 1
                _dihedralFrom = int(tokens[5]) - 1

                # positions of reference atoms
                p_bond = numpy.array(_coords[_bondedTo])
                p_angle = numpy.array(_coords[_angleFrom])
                p_dihedral = numpy.array(_coords[_dihedralFrom])

                # define local coordinate system
                v1 = p_bond - p_angle
                v2 = p_dihedral - p_angle

                # normalize
                v1 = v1 / numpy.linalg.norm(v1)
                v2 = v2 / numpy.linalg.norm(v2)

                # normal to plane defined by v1 and v2
                n = numpy.cross(v1, v2)
                n = n / numpy.linalg.norm(n)

                # perpendicular vector in plane
                m = numpy.cross(n, v1)

                # convert angles to radians
                theta = numpy.radians(_angle)
                phi = numpy.radians(_dihedral)

                # build new coordinate relative to bonded atom
                new_coord = (p_bond + _distance * (numpy.cos(theta) * (-v1) +
                             numpy.sin(theta) * (numpy.cos(phi) * m + numpy.sin(phi) * n)))

                _coords.append(new_coord.tolist())

        self.coords = numpy.array(_coords)
        self.coords_internal = self.coords.flatten()
