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
This module contains the :class:`QutritDevice` abstract base class.
"""

# For now, arguments may be different from the signatures provided in Device
# e.g. instead of expval(self, observable, wires, par) have expval(self, observable)
# pylint: disable=arguments-differ, abstract-method, no-value-for-parameter,too-many-instance-attributes,too-many-branches, no-member, bad-option-value, arguments-renamed
import itertools

import numpy as np

import pennylane as qml
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
from pennylane import QubitDevice
from pennylane.wires import Wires   # pylint: disable=unused-import


class QutritDevice(QubitDevice):  # pylint: disable=too-many-public-methods
    """Abstract base class for Pennylane qutrit devices.

    The following abstract method **must** be defined:

    * :meth:`~.apply`: append circuit operations, compile the circuit (if applicable),
      and perform the quantum computation.

    Devices that generate their own samples (such as hardware) may optionally
    overwrite :meth:`~.probabilty`. This method otherwise automatically
    computes the probabilities from the generated samples, and **must**
    overwrite the following method:

    * :meth:`~.generate_samples`: Generate samples from the device from the
      exact or approximate probability distribution.

    Analytic devices **must** overwrite the following method:

    * :meth:`~.analytic_probability`: returns the probability or marginal probability from the
      device after circuit execution. :meth:`~.marginal_prob` may be used here.

    This device contains common utility methods for qutrit-based devices. These
    do not need to be overwritten. Utility methods include:

    * :meth:`~.expval`, :meth:`~.var`, :meth:`~.sample`: return expectation values,
      variances, and samples of observables after the circuit has been rotated
      into the observable eigenbasis.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        r_dtype: Real floating point precision type.
        c_dtype: Complex floating point precision type.
    """

    # TODO: Update set of supported observables as new observables are added
    observables = {
        "Identity",
    }

    @classmethod
    def capabilities(cls):

        capabilities = super().capabilities().copy()
        capabilities.update(model="qutrit")
        return capabilities

    # Added here for tests to pass (See tests.test_qutrit_device.ExtractStatistics.test_results_no_state())
    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def statistics(self, observables, shot_range=None, bin_size=None):
        # Overloading QubitDevice.statistics() as VnEntropy and MutualInfo not yet supported for QutritDevice
        results = []

        for obs in observables:
            # Pass instances directly
            if obs.return_type is Expectation:
                results.append(self.expval(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Variance:
                results.append(self.var(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Sample:
                results.append(self.sample(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Probability:
                results.append(
                    self.probability(wires=obs.wires, shot_range=shot_range, bin_size=bin_size)
                )

            elif obs.return_type is State:
                if len(observables) > 1:
                    raise qml.QuantumFunctionError(
                        "The state or density matrix cannot be returned in combination"
                        " with other return types"
                    )
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError(
                        "Returning the state is not supported when using custom wire labels"
                    )
                # Check if the state is accessible and decide to return the state or the density
                # matrix.
                results.append(self.access_state(wires=obs.wires))

            elif obs.return_type is not None:
                raise qml.QuantumFunctionError(
                    f"Unsupported return type specified for observable {obs.name}"
                )

            # TODO: Add support for MutualInfo and VnEntropy return types. Currently, qml.math is hard coded
            # to calculate these quantities for qubit states, so it needs to be updated before these return types
            # can be supported for qutrits.

        return results

    def generate_samples(self):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant trit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        number_of_states = 3**self.num_wires

        rotated_prob = self.analytic_probability()

        samples = self.sample_basis_states(number_of_states, rotated_prob)
        return QutritDevice.states_to_ternary(samples, self.num_wires)

    @staticmethod
    def generate_basis_states(num_wires, dtype=np.uint32):
        """
        Generates basis states in ternary representation according to the number
        of wires specified.

        Args:
            num_wires (int): the number wires
            dtype=np.uint32 (type): the data type of the arrays to use

        Returns:
            array[int]: the sampled basis states
        """
        # A slower, but less memory intensive method
        basis_states_generator = itertools.product((0, 1, 2), repeat=num_wires)
        return np.fromiter(itertools.chain(*basis_states_generator), dtype=dtype).reshape(
            -1, num_wires
        )

    @staticmethod
    def states_to_ternary(samples, num_wires, dtype=np.int64):
        """Convert basis states from base 10 to ternary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (array[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qutrits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            array[int]: basis states in ternary representation
        """
        ternary_arr = []
        for sample in samples:
            num = []
            for _ in range(num_wires):
                sample, r = divmod(sample, 3)
                num.append(r)

            ternary_arr.append(num[::-1])

        return np.array(ternary_arr, dtype=dtype)

    def density_matrix(self, wires):
        """Returns the reduced density matrix prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        # TODO: Add density matrix support. Currently, qml.math is hard-coded to work only with qubit states,
        # so it needs to be updated to be able to handle calculations for qutrits before this method can be
        # implemented.
        raise NotImplementedError

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """

        wires = wires or self.wires
        # convert to a wires object
        wires = Wires(wires)
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)
        samples = self._samples[sample_slice, device_wires]

        # convert samples from a list of 0, 1 integers, to base 10 representation
        powers_of_three = 3 ** np.arange(len(device_wires))[::-1]
        indices = samples @ powers_of_three

        # count the basis state occurrences, and construct the probability vector
        if bin_size is not None:
            bins = len(samples) // bin_size

            indices = indices.reshape((bins, -1))
            prob = np.zeros([3 ** len(device_wires), bins], dtype=np.float64)

            # count the basis state occurrences, and construct the probability vector
            for b, idx in enumerate(indices):
                basis_states, counts = np.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size

        else:
            basis_states, counts = np.unique(indices, return_counts=True)
            prob = np.zeros([3 ** len(device_wires)], dtype=np.float64)
            prob[basis_states] = counts / len(samples)

        return self._asarray(prob, dtype=self.R_DTYPE)

    def marginal_prob(self, prob, wires=None):
        r"""Return the marginal probability of the computational basis
        states by summing the probabiliites on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            If the provided wires are not in the order as they appear on the device,
            the returned marginal probabilities take this permutation into account.

            For example, if the addressable wires on this device are ``Wires([0, 1, 2])`` and
            this function gets passed ``wires=[2, 0]``, then the returned marginal
            probability vector will take this 'reversal' of the two wires
            into account:

            .. math::

                \mathbb{P}^{(2, 0)}
                            = \left[
                               |00\rangle, |10\rangle, |01\rangle, |11\rangle
                              \right]

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array of the resulting marginal probabilities.
        """

        if wires is None:
            # no need to marginalize
            return prob

        wires = Wires(wires)
        # determine which subsystems are to be summed over
        inactive_wires = Wires.unique_wires([self.wires, wires])

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        inactive_device_wires = self.map_wires(inactive_wires)

        # reshape the probability so that each axis corresponds to a wire
        prob = self._reshape(prob, [3] * self.num_wires)

        # sum over all inactive wires
        # hotfix to catch when default.qubit uses this method
        # since then device_wires is a list
        if isinstance(inactive_device_wires, Wires):
            prob = self._flatten(self._reduce_sum(prob, inactive_device_wires.labels))
        else:
            prob = self._flatten(self._reduce_sum(prob, inactive_device_wires))

        # The wires provided might not be in consecutive order (i.e., wires might be [2, 0]).
        # If this is the case, we must permute the marginalized probability so that
        # it corresponds to the orders of the wires passed.
        num_wires = len(device_wires)
        basis_states = self.generate_basis_states(num_wires)
        basis_states = basis_states[:, np.argsort(np.argsort(device_wires))]

        powers_of_three = 3 ** np.arange(len(device_wires))[::-1]
        perm = basis_states @ powers_of_three
        return self._gather(prob, perm)

    def sample(self, observable, shot_range=None, bin_size=None):

        # TODO: Add special cases for any observables that require them once list of
        # observables is updated.

        # translate to wire labels used by device
        device_wires = self.map_wires(observable.wires)
        name = observable.name  # pylint: disable=unused-variable
        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)

        # Replace the basis state in the computational basis with the correct eigenvalue.
        # Extract only the columns of the basis samples required based on ``wires``.
        samples = self._samples[
            sample_slice, np.array(device_wires, dtype=np.int32)
        ]  # Add np.array here for Jax support.
        powers_of_three = 3 ** np.arange(samples.shape[-1])[::-1]
        indices = samples @ powers_of_three
        indices = np.array(indices)  # Add np.array here for Jax support.
        try:
            samples = observable.eigvals()[indices]
        except qml.operation.EigvalsUndefinedError as e:
            # if observable has no info on eigenvalues, we cannot return this measurement
            raise qml.operation.EigvalsUndefinedError(
                f"Cannot compute samples of {observable.name}."
            ) from e

        if bin_size is None:
            return samples

        return samples.reshape((bin_size, -1))

    # TODO: Implement function
    def adjoint_jacobian(
        self, tape, starting_state=None, use_device_state=False
    ):  # pylint: disable=missing-function-docstring
        raise NotImplementedError
