# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule contains the ternary discrete-variable quantum operations
that do not depend on any parameters.
"""
# pylint:disable=arguments-differ
import numpy as np

import pennylane as qml  # pylint: disable=unused-import
from pennylane.operation import Operation
from pennylane.wires import Wires

OMEGA = np.exp(2 * np.pi * 1j / 3)
ZETA = OMEGA ** (1 / 3)

# Note: When Operation.matrix() is used for qutrit operations, `wire_order` must be `None` as specifying
# an order that is expected to return a permuted and expanded matrix different from the canonical matrix
# will lead to errors as `expand_matrix()` in `pennylane/operation.py`, which is used to compute the
# permuted and expanded matrix from the canonical matrix, is hard coded to work correctly for qubits.


class TShift(Operation):
    r"""TShift(wires)
    The qutrit shift operator

    .. math:: TShift = \begin{bmatrix}
                        0 & 0 & 1 \\
                        1 & 0 & 0 \\
                        0 & 1 & 0
                    \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TShift.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TShift.compute_matrix())
        [[0 0 1]
         [1 0 0]
         [0 1 0]]
        """
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TShift.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TShift.compute_eigvals())
        [ -0.5+0.8660254j -0.5-0.8660254j  1. +0.j       ]
        """
        return np.array([OMEGA, OMEGA**2, 1])

    # TODO: Add compute_decomposition once parametric ops are added.

    def pow(self, z):
        if isinstance(z, int):
            z_mod3 = z % 3
            if z_mod3 < 2:
                return super().pow(z_mod3)
            return [self.adjoint()]
        return super().pow(z)

    def adjoint(self):
        op = TShift(wires=self.wires)
        op.inverse = not self.inverse
        return op


class TClock(Operation):
    r"""TClock(wires)
    Ternary Clock gate

    .. math:: TClock = \begin{bmatrix}
                        1 & 0      & 0        \\
                        0 & \omega & 0        \\
                        0 & 0      & \omega^2
                    \end{bmatrix}
                    \omega = \exp{2 \cdot \pi \cdot i / 3}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TClock.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TClock.compute_matrix())
        [[ 1. +0.j         0. +0.j         0. +0.j       ]
         [ 0. +0.j        -0.5+0.8660254j  0. +0.j       ]
         [ 0. +0.j         0. +0.j        -0.5-0.8660254j]]
        """
        return np.diag([1, OMEGA, OMEGA**2])

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TClock.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TClock.compute_eigvals())
        [ 1. +0.j        -0.5+0.8660254j -0.5-0.8660254j]
        """
        return np.array([1, OMEGA, OMEGA**2])

    # TODO: Add compute_decomposition() once parametric ops are added.

    def pow(self, z):
        if isinstance(z, int):
            z_mod3 = z % 3
            if z_mod3 < 2:
                return super().pow(z_mod3)
            return [self.adjoint()]
        return super().pow(z)

    def adjoint(self):
        op = TClock(wires=self.wires)
        op.inverse = not self.inverse
        return op


class TAdd(Operation):
    r"""TAdd(wires)
    The 2-qutrit controlled add gate

    .. math:: TAdd = \begin{bmatrix}
                        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
                    \end{bmatrix}

    .. note:: The first wire provided corresponds to the **control qutrit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TAdd.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TAdd.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0 0]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.
        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.TAdd.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.TAdd.compute_eigvals())
        [-0.5+0.8660254j -0.5-0.8660254j  1. +0.j        -0.5+0.8660254j -0.5-0.8660254j  1. +0.j         1. +0.j         1. +0.j         1. +0.j       ]
        """
        return np.array([OMEGA, OMEGA**2, 1, OMEGA, OMEGA**2, 1, 1, 1, 1])

    # TODO: Add compute_decomposition() once parametric ops are added.

    def pow(self, z):
        if isinstance(z, int):
            z_mod3 = z % 3
            if z_mod3 < 2:
                return super().pow(z_mod3)
            return [self.adjoint()]
        return super().pow(z)

    def adjoint(self):
        op = TAdd(self.wires)
        op.inverse = not self.inverse
        return op

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class TSWAP(Operation):
    r"""TSWAP(wires)
    The ternary swap operator

    .. math:: TSWAP = \begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            \end{bmatrix}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "TSWAP"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0 0]
         [0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 0 1 0 0]
         [0 1 0 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 1 0]
         [0 0 1 0 0 0 0 0 0]
         [0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0 0 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    # TODO: Add compute_decomposition()

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return TSWAP(wires=self.wires)


class TX(Operation):
    r"""TX(wires, subspace)
    Ternary X operation

    Performs the Pauli X operation on the specified 2D subspace. The subspace is
    given as a keyword argument and determines which two of three single-qutrit
    basis states the operation applies to.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
        subspace (Sequence[int]): the 2D subspace on which to apply operation
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)

    **Example**

    The specified subspace will determine which basis states the operation actually
    applies to:

    >>> qml.TX(wires=0, subspace=[0, 1]).matrix()

    >>> qml.TX(wires=0, subspace=[0, 2]).matrix()

    >>> qml.TX(wires=0, subspace=[1, 2]).matrix()
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "TX"

    def __init__(
        self, wires, subspace=[0, 1], do_queue=True
    ):  # pylint: disable=dangerous-default-value
        self._subspace = subspace
        self._hyperparameters = {
            "subspace": self.subspace,
        }
        super().__init__(wires=wires, do_queue=do_queue)

    @property
    def subspace(self):
        """The single-qutrit basis states which the operator acts on

        This property returns the 2D subspace on which the operator acts. This subspace
        determines which two single-qutrit basis states the operator acts on. The remaining
        basis state is not affected by the operator.

        Returns:
            tuple[int]: subspace on which operator acts
        """
        return tuple(sorted(self._subspace))

    @staticmethod
    def compute_matrix(subspace=[0, 1]):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.TX.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.TX.compute_matrix(subspace=[0, 2]))
        [[0 0 1]
         [0 1 0]
         [1 0 0]]
        """

        if len(subspace) != 2:
            raise ValueError(
                "subspace must be a sequence with 2 unique elements from the set {0, 1, 2}"
            )

        if subspace[0] == subspace[1]:
            raise ValueError("Elements of subspace list must be unique.")

        if not all([s in {0, 1, 2} for s in subspace]):
            raise ValueError("Subspace elements should be 0, 1, or 2.")

        subspace = tuple(sorted(subspace))

        mat = np.zeros((3, 3))

        unused_ind = int({0, 1, 2}.difference(set(subspace)))
        mat[subspace[0], subspace[1]] = 1
        mat[subspace[1], subspace[0]] = 1
        mat[unused_ind, unused_ind] = 1

        return mat
