import abc
import numpy
import typing


class BaseGeometry(abc.ABC):
    """ 
    Abstract Base Class (ABC) for molecular geometry representations. 
    This class enforces the implementation of a `geometry` property in subclasses.
    Subclasses must define how molecular geometry information is stored and accessed.
    """

    def __init__(self) -> None:
        """
        Initialize the base geometry object with default molecular attributes.

        This constructor sets up placeholders and default values for storing
        molecular geometry information. Subclasses such as Cartesian and ZMatrix
        are expected to populate these attributes with meaningful data.

        Attributes:
            atoms (list or None):
                A container for atomic symbols or atom objects. Initially set to None
                until defined by subclasses or external input.

            coords (numpy.matrix):
                A matrix of Cartesian coordinates with shape (0, 3). Each row
                corresponds to an atom, with columns representing x, y, and z
                coordinates. Initialized as an empty matrix.

            atomicnumbers (numpy.ndarray):
                A 1D array of atomic numbers corresponding to the atoms in the system.
                Initialized as a zero-length array.

            coords_internal (numpy.ndarray):
                A flattened representation of `coords`, useful for internal
                computations or interfacing with algorithms that require a
                1D coordinate vector.

            natoms (int or None):
                The number of atoms in the system. Initially None until atoms are
                defined.

            charge (int):
                The total molecular charge. Defaults to 0 (neutral molecule).

            multi (int):
                The spin multiplicity of the system. Defaults to 1 (singlet state).

        Notes:
            - This base initialization provides a consistent structure for
            molecular geometry classes.
            - Subclasses should validate and populate these attributes with
            chemically meaningful values.
        """
        self.atoms = []
        self.coords = numpy.matrix((0, 3))
        self.atomicnumbers = numpy.zeros(0)
        self.coords_internal = self.coords.flatten()
        self.natoms = None
        self.charge = 0
        self.multi = 1

    @property
    @abc.abstractmethod
    def geometry(self) -> typing.Any:
        """ 
        Abstract property representing the molecular geometry. 
        Returns: 
        Any: The internal representation of the geometry. 
        """
        pass

    @geometry.setter
    @abc.abstractmethod
    def geometry(self, value: typing.Any) -> None:
        """ 
        Abstract setter for the molecular geometry. 
        Args: 
        value (Any): The new geometry representation to assign. 
        """
        pass
