r"""
The default.qutrit device is Pennylane's standard qutrit-based device.

It implements the :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qutrit operations <pennylane.ops.qutrit>`, and provides a simple pure state
simulation of qutrit-based quantum circuit architecture
"""
import functools, itertools
import numpy as np

import pennylane as qml
from pennylane import QutritDevice, DeviceError
from pennylane.wires import Wires, WireError
from .._version import __version__
from pennylane.devices.default_qubit import _get_slice

from pennylane.measurements import MeasurementProcess

# tolerance for numerical errors
tolerance = 1e-10

OMEGA = np.exp(2 * np.pi * 1j / 3)


class DefaultQutrit(QutritDevice):
    """Default qutrit device for PennyLane

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """

    name = "Default qutrit PennyLane plugin"
    short_name = "default.qutrit"
    pennylane_requires = __version__
    version = __version__
    author = "Mudit Pandey"

    # TODO: Add list of operations and observables
    operations = {
        "QutritUnitary",
        "ControlledQutritUnitary",
        "TShift",
        "TClock",
        "TAdd",
        "TSWAP",
        "Identity",
    }

    observables = {
        "Identity",
        "QutritUnitary"
    }

    def __init__(
        self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None

        # Create the initial state. Internally, we store the
        # state as an array of dimension [3]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # TODO: Add operations
        self._apply_ops = {
            "TShift": self._apply_tshift,
            "TClock": self._apply_tclock,
            "TSWAP": self._apply_tswap,
            "TAdd": self._apply_tadd,
        }

    @functools.lru_cache()
    def map_wires(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        try:
            mapped_wires = [self.wire_map[w] for w in wires]
        except KeyError as e:
            raise WireError(
                f"Did not find some of the wires {wires.labels} on device with wires {self.wires.labels}."
            ) from e

        return mapped_wires

    def define_wire_map(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        consecutive_wires = range(self.num_wires)
        wire_map = zip(wires, consecutive_wires)
        return dict(wire_map)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            # if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
            #     raise DeviceError(
            #         f"Operation {operation.name} cannot be used after other Operations have already been applied "
            #         f"on a {self.short_name} device."
            #     )

            # if isinstance(operation, QubitStateVector):
            #     self._apply_state_vector(operation.parameters[0], operation.wires)
            # elif isinstance(operation, BasisState):
            #     self._apply_basis_state(operation.parameters[0], operation.wires)
            # elif isinstance(operation, Snapshot):
            #     if self._debugger and self._debugger.active:
            #         state_vector = np.array(self._flatten(self._state))
            #         if operation.tag:
            #             self._debugger.snapshots[operation.tag] = state_vector
            #         else:
            #             self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
            if False:
                # DO NOTHING
                continue
            else:
                self._state = self._apply_operation(self._state, operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._state = self._apply_operation(self._state, operation)

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        if operation.base_name == "Identity":
            return state
        wires = operation.wires

        if operation.base_name in self._apply_ops:
            axes = self.wires.indices(wires)
            return self._apply_ops[operation.base_name](state, axes, inverse=operation.inverse)

        matrix = self._asarray(self._get_unitary_matrix(operation), dtype=self.C_DTYPE)

        # if len(wires) <= 2:
        #     # Einsum is faster for small gates
        #     return self._apply_unitary_einsum(state, matrix, wires)

        return self._apply_unitary(state, matrix, wires)

    def _apply_tshift(self, state, axes, inverse=False):
        """Applies a Shift gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|2 \rangle` is transformed to
        :math:`|0\rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        shift = -1 if inverse else 1
        return self._roll(state, shift, axes[0])

    def _apply_tclock(self, state, axes, inverse=False):
        """Applies a ternary Clock gate by adding a phase of :math:`\omega` to the 1 index and
        :math:`\omega^{2}` to the 2 index along the axis specified in ``axes``

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        partial_state = self._apply_phase(state, axes, 1, OMEGA, inverse)
        return self._apply_phase(partial_state, axes, 2, OMEGA**2, inverse)

    def _apply_tswap(self, state, axes, **kwargs):
        """Applies a ternary SWAP gate by performing a partial transposition along the
        specified axes.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        all_axes = list(range(len(state.shape)))
        all_axes[axes[0]] = axes[1]
        all_axes[axes[1]] = axes[0]
        return self._transpose(state, all_axes)

    def _apply_tadd(self, state, axes, **kwargs):
        """Applies a controlled add gate by slicing along the first axis specified in ``axes`` and
        applying a TShift transformation along the second axis

        By slicing along the first axis, we are able to select all of the amplitudes with corresponding
        :math:`|1\rangle` and :math:`|2\rangle` for the control qutrit. This means we just need to apply
        a :class:`~.TShift` gate when slicing along index 1, and a :class:`~.TShift` adjoint gate when
        slicing along index 2

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        slices = [_get_slice(i, axes[0], self.num_wires) for i in range(3)]

        # We will be slicing into the state according to state[slices[1]] and state[slices[2]],
        # giving us all of the amplitudes with a |1> and |2> for the control qutrit. The resulting
        # array has lost an axis relative to state and we need to be careful about the axis we
        # roll. If axes[1] is larger than axes[0], then we need to shift the target axis down by
        # one, otherwise we can leave as-is. For example: a state has [0, 1, 2, 3], control=1,
        # target=3. Then, state[slices[1]] has 3 axes and target=3 now corresponds to the second axis.
        if axes[1] > axes[0]:
            target_axes = [axes[1] - 1]
        else:
            target_axes = [axes[1]]

        state_1 = self._apply_tshift(state[slices[1]], axes=target_axes)
        state_2 = self._apply_tshift(state[slices[2]], axes=target_axes, inverse=True)
        return self._stack([state[slices[0]], state_1, state_2], axis=axes[0])

    def _apply_phase(self, state, axes, index, parameters, inverse=False):
        """Applies a phase onto the specified index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            index (int): target index of axis to apply phase to
            parameters (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
        num_wires = len(state.shape)
        slices = [_get_slice(i, axes[0], num_wires) for i in range(3)]

        phase = self._conj(parameters) if inverse else parameters
        state_slices = [self._const_mul(phase if i == index else 1, state[slices[i]]) for i in range(3)]
        return self._stack(state_slices, axis=axes[0])

    def expval(self, observable, shot_range=None, bin_size=None):
        # TODO: Update later
        return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis
        """
        return unitary.matrix()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qutrit",
            supports_reversible_diff=True,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True
        )
        return capabilities

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[3]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(3**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [3] * self.num_wires)

    @property
    def state(self):
        return self._flatten(self._pre_rotated_state)

    def density_matrix(self, wires):
        """Returns the reduced density matrix of a given set of wires.

        Args:
            wires (Wires): wires of the reduced system.

        Returns:
            array[complex]: complex tensor of shape ``(3 ** len(wires), 3 ** len(wires))``
            representing the reduced density matrix.
        """
        dim = self.num_wires
        state = self._pre_rotated_state

        # Return the full density matrix by using numpy tensor product
        if wires == self.wires:
            density_matrix = self._tensordot(state, self._conj(state), axes=0)
            density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))
            return density_matrix

        complete_system = list(range(0, dim))
        traced_system = [x for x in complete_system if x not in wires.labels]

        # Return the reduced density matrix by using numpy tensor product
        density_matrix = self._tensordot(
            state, self._conj(state), axes=(traced_system, traced_system)
        )
        density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))

        return density_matrix

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length
                ``3**len(wires)``
            device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        n_state_vector = state.shape[0]

        if len(qml.math.shape(state)) != 1 or n_state_vector != 3 ** len(device_wires):
            raise ValueError("State vector must be of length 3**wires.")

        norm = qml.math.linalg.norm(state, ord=2)
        if not qml.math.is_abstract(norm):
            if not qml.math.allclose(norm, 1.0, atol=tolerance):
                raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(device_wires) == self.num_wires and sorted(device_wires) == device_wires:
            # Initialize the entire wires with the state
            self._state = self._reshape(state, [3] * self.num_wires)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1, 2], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((3 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [3] * self.num_wires)

        state = self._scatter(ravelled_indices, state, [3**self.num_wires])
        state = self._reshape(state, [3] * self.num_wires)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s, 1s and 2s.
            wires (Wires): wires that the provided computational state should be initialized on
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1, 2}):
            raise ValueError("BasisState parameter must consist of 0, 1, or 2 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 3 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    # TODO: Edit later
    def state(self):
        return self._flatten(self._state)

    def _apply_unitary(self, state, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        mat = self._cast(self._reshape(mat, [3] * len(device_wires) * 2), dtype=self.C_DTYPE)
        axes = (np.arange(len(device_wires), 2 * len(device_wires)), device_wires)
        tdot = self._tensordot(mat, state, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wires]
        perm = list(device_wires) + unused_idxs
        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        return self._transpose(tdot, inv_perm)

    # TODO: Implement function
    # def _apply_unitary_einsum(self, state, mat, wires):
    #     pass

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        flat_state = self._flatten(self._state)
        real_state = self._real(flat_state)
        imag_state = self._imag(flat_state)
        prob = self.marginal_prob(real_state**2 + imag_state**2, wires)
        return prob

