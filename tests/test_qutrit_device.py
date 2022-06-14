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
Unit tests for the :mod:`pennylane` :class:`QutritDevice` class.
"""
import pytest
import numpy as np
from random import random
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QutritDevice, DeviceError, QuantumFunctionError
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
from pennylane.circuit_graph import CircuitGraph
from pennylane.wires import Wires
from pennylane.tape import QuantumTape
from pennylane.measurements import state


@pytest.fixture(scope="function")
def mock_qutrit_device(monkeypatch):
    """A function to create a mock qutrit device that mocks most of the methods except for e.g. probability()"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])
        m.setattr(QutritDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "apply", lambda self, *args, **kwargs: None)

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


mock_qutrit_device_capabilities = {
    "measurements": "everything",
    "returns_state": True,
}


@pytest.fixture(scope="function")
def mock_qutrit_device_extract_stats(monkeypatch):
    """A function to create a mock device that mocks the methods related to
    statistics (expval, var, sample, probability)"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])
        m.setattr(QutritDevice, "expval", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "var", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "sample", lambda self, *args, **kwargs: 0)
        m.setattr(QutritDevice, "state", 0)
        m.setattr(QutritDevice, "density_matrix", lambda self, wires=None: 0)
        m.setattr(QutritDevice, "probability", lambda self, wires=None, *args, **kwargs: 0)
        m.setattr(QutritDevice, "apply", lambda self, x: x)

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


@pytest.fixture(scope="function")
def mock_qutrit_device_with_original_statistics(monkeypatch):
    """A function to create a mock qutrit device that uses the original statistics related
    methods"""
    with monkeypatch.context() as m:
        m.setattr(QutritDevice, "__abstractmethods__", frozenset())
        m.setattr(QutritDevice, "_capabilities", mock_qutrit_device_capabilities)
        m.setattr(QutritDevice, "short_name", "MockQutritDevice")
        m.setattr(QutritDevice, "operations", ["QutritUnitary", "Identity"])
        m.setattr(QutritDevice, "observables", ["Identity"])

        def get_qutrit_device(wires=1):
            return QutritDevice(wires=wires)

        yield get_qutrit_device


class TestOperations:
    """Tests the logic related to operations"""

    def test_op_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError, match="Cannot access the operation queue outside of the execution context!"
        ):
            dev = mock_qutrit_device()
            dev.op_queue

    def test_op_queue_is_filled_during_execution(
        self, mock_qutrit_device, monkeypatch
    ):
        """Tests that the op_queue is correctly filled when apply is called and that accessing
        op_queue raises no error"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0), qml.QutritUnitary(U, wires=0)]
            observables = [qml.expval(qml.Identity(0))]

        call_history = []

        with monkeypatch.context() as m:
            m.setattr(
                QutritDevice,
                "apply",
                lambda self, x, **kwargs: call_history.extend(x + kwargs.get("rotations", [])),
            )
            m.setattr(QutritDevice, "analytic_probability", lambda *args: None)
            dev = mock_qutrit_device()
            dev.execute(tape)

        assert call_history == queue

        assert len(call_history) == 2
        assert isinstance(call_history[0], qml.QutritUnitary)
        assert call_history[0].wires == Wires([0])

        assert isinstance(call_history[1], qml.QutritUnitary)
        assert call_history[1].wires == Wires([0])

    def test_unsupported_operations_raise_error(self, mock_qutrit_device):
        """Tests that the operations are properly applied and queued"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0), qml.Hadamard(wires=1), qml.QutritUnitary(U, wires=2)]
            observables = [qml.expval(qml.Identity(0)), qml.var(qml.Identity(1))]

        with pytest.raises(DeviceError, match="Gate Hadamard not supported on device"):
            dev = mock_qutrit_device()
            dev.execute(tape)

    unitaries = [unitary_group.rvs(3, random_state=1967) for _ in range(3)]
    numeric_queues = [
        [qml.QutritUnitary(unitaries[0], wires=[0])],
        [
            qml.QutritUnitary(unitaries[0], wires=[0]),
            qml.QutritUnitary(unitaries[1], wires=[1]),
            qml.QutritUnitary(unitaries[2], wires=[2]),
        ],
    ]

    observables = [[qml.Identity(0)], [qml.Identity(1)]]

    @pytest.mark.parametrize("observables", observables)
    @pytest.mark.parametrize("queue", numeric_queues)
    def test_passing_keyword_arguments_to_execute(
        self, mock_qutrit_device, monkeypatch, queue, observables
    ):
        """Tests that passing keyword arguments to execute propagates those kwargs to the apply()
        method"""
        with qml.tape.QuantumTape() as tape:
            for op in queue + observables:
                op.queue()

        call_history = {}

        with monkeypatch.context() as m:
            m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: call_history.update(kwargs))
            dev = mock_qutrit_device()
            dev.execute(tape, hash=tape.graph.hash)

        len(call_history.items()) == 1
        call_history["hash"] = tape.graph.hash


class TestObservables:
    """Tests the logic related to observables"""

    U = unitary_group.rvs(3, random_state=10)

    def test_obs_queue_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to op_queue outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the observable value queue outside of the execution context!",
        ):
            dev = mock_qutrit_device()
            dev.obs_queue

    def test_unsupported_observables_raise_error(self, mock_qutrit_device):
        """Tests that the operations are properly applied and queued"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            queue = [qml.QutritUnitary(U, wires=0)]
            observables = [qml.expval(qml.Hadamard(0))]

        with pytest.raises(DeviceError, match="Observable Hadamard not supported on device"):
            dev = mock_qutrit_device()
            dev.execute(tape)

    def test_unsupported_observable_return_type_raise_error(
        self, mock_qutrit_device, monkeypatch
    ):
        """Check that an error is raised if the return type of an observable is unsupported"""
        U = unitary_group.rvs(3, random_state=10)

        with qml.tape.QuantumTape() as tape:
            qml.QutritUnitary(U, wires=0)
            qml.measurements.MeasurementProcess(
                return_type="SomeUnsupportedReturnType", obs=qml.Identity(0)
            )

        with monkeypatch.context() as m:
            m.setattr(QutritDevice, "apply", lambda self, x, **kwargs: None)
            with pytest.raises(
                qml.QuantumFunctionError, match="Unsupported return type specified for observable"
            ):
                dev = mock_qutrit_device()
                dev.execute(tape)


class TestParameters:
    """Test for checking device parameter mappings"""

    def test_parameters_accessed_outside_execution_context(self, mock_qutrit_device):
        """Tests that a call to parameters outside the execution context raises the correct error"""

        with pytest.raises(
            ValueError,
            match="Cannot access the free parameter mapping outside of the execution context!",
        ):
            dev = mock_qutrit_device()
            dev.parameters


class TestExtractStatistics:
    @pytest.mark.parametrize("returntype", [Expectation, Variance, Sample, Probability, State])
    def test_results_created(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method simply builds a results list without any side-effects"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qutrit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == [0]

    def test_results_no_state(self, mock_qutrit_device_extract_stats, monkeypatch):
        """Tests that the statistics method raises an AttributeError when a State return type is
        requested when QutritDevice does not have a state attribute"""
        with monkeypatch.context():
            dev = mock_qutrit_device_extract_stats()
            delattr(dev.__class__, "state")
            with pytest.raises(
                qml.QuantumFunctionError, match="The state is not available in the current"
            ):
                dev.statistics([state()])

    @pytest.mark.parametrize("returntype", [None])
    def test_results_created_empty(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method returns an empty list if the return type is None"""

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with monkeypatch.context() as m:
            dev = mock_qutrit_device_extract_stats()
            results = dev.statistics([obs])

        assert results == []

    @pytest.mark.parametrize("returntype", ["not None"])
    def test_error_return_type_none(self, mock_qutrit_device_extract_stats, monkeypatch, returntype):
        """Tests that the statistics method raises an error if the return type is not well-defined and is not None"""

        assert returntype not in [Expectation, Variance, Sample, Probability, State, None]

        class SomeObservable(qml.operation.Observable):
            num_wires = 1
            return_type = returntype

        obs = SomeObservable(wires=0)

        with pytest.raises(qml.QuantumFunctionError, match="Unsupported return type"):
            dev = mock_qutrit_device_extract_stats()
            dev.statistics([obs])


class TestGenerateSamples:
    """Test the generate_samples method"""

    def test_auxiliary_methods_called_correctly(self, mock_qutrit_device, monkeypatch):
        """Tests that the generate_samples method calls on its auxiliary methods correctly"""

        dev = mock_qutrit_device()
        number_of_states = 3**dev.num_wires

        with monkeypatch.context() as m:
            # Mock the auxiliary methods such that they return the expected values
            m.setattr(QutritDevice, "sample_basis_states", lambda self, wires, b: wires)
            m.setattr(QutritDevice, "states_to_ternary", lambda a, b: (a, b))
            m.setattr(QutritDevice, "analytic_probability", lambda *args: None)
            m.setattr(QutritDevice, "shots", 1000)
            dev._samples = dev.generate_samples()

        assert dev._samples == (number_of_states, dev.num_wires)


class TestSampleBasisStates:
    """Test the sample_basis_states method"""

    def test_sampling_with_correct_arguments(self, mock_qutrit_device, monkeypatch):
        """Tests that the sample_basis_states method samples with the correct arguments"""

        shots = 1000

        number_of_states = 9
        dev = mock_qutrit_device()
        dev.shots = shots
        state_probs = [0.1] * 9
        state_probs[0] = 0.2

        with monkeypatch.context() as m:
            # Mock the numpy.random.choice method such that it returns the expected values
            m.setattr("numpy.random.choice", lambda x, y, p: (x, y, p))
            res = dev.sample_basis_states(number_of_states, state_probs)

        assert np.array_equal(res[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
        assert res[1] == shots
        assert res[2] == state_probs

    def test_raises_deprecation_warning(self, mock_qutrit_device, monkeypatch):
        """Test that sampling basis states on a device with shots=None produces a warning."""

        dev = mock_qutrit_device()
        number_of_states = 9
        dev.shots = None
        state_probs = [0.1] * 9
        state_probs[0] = 0.2

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device",
        ):
            dev.sample_basis_states(number_of_states, state_probs)


class TestStatesToTernary:
    """Test the states_to_ternary method"""

    def test_correct_conversion_two_states(self, mock_qutrit_device):
        """Tests that the sample_basis_states method converts samples to binary correctly"""
        wires = 4
        shots = 10

        number_of_states = 3**wires
        basis_states = np.arange(number_of_states)
        samples = np.random.choice(basis_states, shots)

        dev = mock_qutrit_device()
        res = dev.states_to_ternary(samples, wires)

        expected = []

        for s in samples:
            num = []
            for _ in range(wires):
                num.append(s % 3)
                s = s // 3

            expected.append(num[::-1])

        assert np.array_equal(res, np.array(expected))

    test_ternary_conversion_data = [
        (
            np.array([2, 3, 2, 0, 0, 1, 6, 8, 5, 6]),
            np.array(
                [
                    [0, 2],
                    [1, 0],
                    [0, 2],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [2, 0],
                    [2, 2],
                    [1, 2],
                    [2, 0],
                ]
            ),
        ),
        (
            np.array([2, 7, 6, 8, 4, 1, 5]),
            np.array(
                [
                    [0, 2],
                    [2, 1],
                    [2, 0],
                    [2, 2],
                    [1, 1],
                    [0, 1],
                    [1, 2],
                ]
            ),
        ),
        (
            np.array([10, 7, 2, 15, 26, 20, 18, 24, 11, 6, 1, 0]),
            np.array(
                [
                    [1, 0, 1],
                    [0, 2, 1],
                    [0, 0, 2],
                    [1, 2, 0],
                    [2, 2, 2],
                    [2, 0, 2],
                    [2, 0, 0],
                    [2, 2, 0],
                    [1, 0, 2],
                    [0, 2, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        ),
    ]

    @pytest.mark.parametrize("samples, ternary_states", test_ternary_conversion_data)
    def test_correct_conversion(self, mock_qutrit_device, samples, ternary_states, tol):
        """Tests that the states_to_binary method converts samples to binary correctly"""
        dev = mock_qutrit_device()
        dev.shots = 5
        wires = ternary_states.shape[1]
        res = dev.states_to_ternary(samples, wires)
        assert np.allclose(res, ternary_states, atol=tol, rtol=0)


# TODO: Add tests for expval, var after observables are added
# class TestExpval:
#     pass


# class TestVar:
#     pass


# class TestSample:
#     pass


class TestEstimateProb:
    """Test the estimate_probability method"""

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0], [0.5, 0.25, 0.25]),
            (None, [0.25, 0, 0.25, 0, 0.25, 0, 0, 0, 0.25]),
            ([0, 1], [0.25, 0, 0.25, 0, 0.25, 0, 0, 0, 0.25]),
            ([1], [0.25, 0.25, 0.5]),
        ]
    )
    def test_estimate_probability(
        self, wires, expected, mock_qutrit_device_with_original_statistics, monkeypatch
    ):
        """Tests probability method when the analytic attribute is True."""
        dev = mock_qutrit_device_with_original_statistics(wires=2)
        samples = np.array([[0, 0], [2, 2], [1, 1], [0, 2]])

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            m.setattr(dev, "shots", 4)
            res = dev.estimate_probability(wires=wires)

        assert np.allclose(res, expected)


class TestMarginalProb:
    """Test the marginal_prob method"""

    @pytest.mark.parametrize(
        "wires, inactive_wires",
        [
            ([0], [1, 2]),
            ([1], [0, 2]),
            ([2], [0, 1]),
            ([0, 1], [2]),
            ([0, 2], [1]),
            ([1, 2], [0]),
            ([0, 1, 2], []),
            (Wires([0]), [1, 2]),
            (Wires([0, 1]), [2]),
            (Wires([0, 1, 2]), []),
        ],
    )
    def test_correct_arguments_for_marginals(
        self, mock_qutrit_device_with_original_statistics, mocker, wires, inactive_wires, tol
    ):
        """Test that the correct arguments are passed to the marginal_prob method"""

        # Generate probabilities
        probs = np.array([random() for i in range(3**3)])
        probs /= sum(probs)

        spy = mocker.spy(np, "sum")
        dev = mock_qutrit_device_with_original_statistics(wires=3)
        res = dev.marginal_prob(probs, wires=wires)
        array_call = spy.call_args[0][0]
        axis_call = spy.call_args[1]["axis"]

        assert np.allclose(array_call.flatten(), probs, atol=tol, rtol=0)
        assert axis_call == tuple(inactive_wires)

    p = np.arange(0.01, 0.28, 0.01) / np.sum(np.arange(0.01, 0.28, 0.01))
    probs = np.reshape(p, [3] * 3)
    s00 = np.sum(probs[0, :, 0])
    s10 = np.sum(probs[1, :, 0])
    s20 = np.sum(probs[2, :, 0])
    s01 = np.sum(probs[0, :, 1])
    s11 = np.sum(probs[1, :, 1])
    s21 = np.sum(probs[2, :, 1])
    s02 = np.sum(probs[0, :, 2])
    s12 = np.sum(probs[1, :, 2])
    s22 = np.sum(probs[2, :, 2])
    m_probs = np.array([s00, s10, s20, s01, s11, s21, s02, s12, s22])

    marginal_test_data = [
        (
            np.array([0.1, 0.2, 0.3, 0.04, 0.03, 0.02, 0.01, 0.18, 0.12]),
            np.array([0.15, 0.41, 0.44]),
            [1],
        ),
        (
            np.array([0.1, 0.2, 0.3, 0.04, 0.03, 0.02, 0.01, 0.18, 0.12]),
            np.array([0.6, 0.09, 0.31]),
            [0],
        ),
        (p, m_probs, [2, 0]),
    ]

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned(
        self, mock_qutrit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that the correct marginals are returned by the marginal_prob method"""
        num_wires = int(np.log(len(probs)) / np.log(3))     # Same as log_3(len(probs))
        dev = mock_qutrit_device_with_original_statistics(num_wires)
        res = dev.marginal_prob(probs, wires=wires)
        assert np.allclose(res, marginals, atol=tol, rtol=0)

    @pytest.mark.parametrize("probs, marginals, wires", marginal_test_data)
    def test_correct_marginals_returned_wires_none(
        self, mock_qutrit_device_with_original_statistics, probs, marginals, wires, tol
    ):
        """Test that passing wires=None simply returns the original probability."""
        num_wires = int(np.log(len(probs)) / np.log(3))     # Same as log_3(len(probs))
        dev = mock_qutrit_device_with_original_statistics(wires=num_wires)
        dev.num_wires = num_wires

        res = dev.marginal_prob(probs, wires=None)
        assert np.allclose(res, probs, atol=tol, rtol=0)


class TestActiveWires:
    """Test that the active_wires static method works as required."""

    def test_active_wires_from_queue(self, mock_qutrit_device):
        queue = [qml.QutritUnitary(np.eye(9), wires=[0, 2]), qml.QutritUnitary(np.eye(3), wires=0), qml.expval(qml.Identity(wires=5))]

        dev = mock_qutrit_device(wires=6)
        res = dev.active_wires(queue)

        assert res == Wires([0, 2, 5])


class TestCapabilities:
    """Test that a default qutrit device defines capabilities that all devices inheriting
    from it will automatically have."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""
        capabilities = {
            "model": "qutrit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "supports_broadcasting": False,
        }
        assert capabilities == QutritDevice.capabilities()


class TestExecution:
    """Tests for the execute method"""

    def test_device_executions(self):
        """Test the number of times a qubit device is executed over a QNode's
        lifetime is tracked by `num_executions`"""

        dev_1 = qml.device("default.qutrit", wires=2)

        def circuit_1(U1, U2, U3):
            qml.QutritUnitary(U1, wires=[0])
            qml.QutritUnitary(U2, wires=[1])
            qml.QutritUnitary(U3, wires=[0, 1])
            return qml.state()

        node_1 = qml.QNode(circuit_1, dev_1)
        num_evals_1 = 10

        for _ in range(num_evals_1):
            node_1(np.eye(3), np.eye(3), np.eye(9))
        assert dev_1.num_executions == num_evals_1

        # test a second instance of a default qubit device
        dev_2 = qml.device("default.qutrit", wires=2)

        def circuit_2(U1, U2):
            qml.QutritUnitary(U1, wires=[0])
            qml.QutritUnitary(U2, wires=[1])
            return qml.state()

        node_2 = qml.QNode(circuit_2, dev_2)
        num_evals_2 = 5

        for _ in range(num_evals_2):
            node_2(np.eye(3), np.eye(3))
        assert dev_2.num_executions == num_evals_2

        # test a new circuit on an existing instance of a qubit device
        def circuit_3(U1, U2):
            qml.QutritUnitary(U1, wires=[0])
            qml.QutritUnitary(U2, wires=[0, 1])
            return qml.state()

        node_3 = qml.QNode(circuit_3, dev_1)
        num_evals_3 = 7

        for _ in range(num_evals_3):
            node_3(np.eye(3), np.eye(9))
        assert dev_1.num_executions == num_evals_1 + num_evals_3


class TestBatchExecution:
    """Tests for the batch_execute method."""

    with qml.tape.QuantumTape() as tape1:
        qml.QutritUnitary(np.eye(3), wires=0)
        qml.expval(qml.Identity(0)), qml.expval(qml.Identity(1))

    with qml.tape.QuantumTape() as tape2:
        qml.QutritUnitary(np.eye(3), wires=0)
        qml.expval(qml.Identity(0))

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_execute(self, n_tapes, mocker, mock_qutrit_device):
        """Tests that the device's execute method is called the correct number of times."""

        dev = mock_qutrit_device(wires=2)
        spy = mocker.spy(QutritDevice, "execute")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_reset(self, n_tapes, mocker, mock_qutrit_device):
        """Tests that the device's reset method is called the correct number of times."""

        dev = mock_qutrit_device(wires=2)

        spy = mocker.spy(QutritDevice, "reset")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_result(self, mock_qutrit_device, r_dtype, tol):
        """Tests that the result has the correct shape and entry types."""

        dev = mock_qutrit_device(wires=2)
        dev.R_DTYPE = r_dtype

        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)

        assert len(res) == 2
        assert np.allclose(res[0], dev.execute(self.tape1), rtol=tol, atol=0)
        assert np.allclose(res[1], dev.execute(self.tape2), rtol=tol, atol=0)
        assert res[0].dtype == r_dtype
        assert res[1].dtype == r_dtype

    def test_result_empty_tape(self, mock_qutrit_device, tol):
        """Tests that the result has the correct shape and entry types for empty tapes."""

        dev = mock_qutrit_device(wires=2)

        empty_tape = qml.tape.QuantumTape()
        tapes = [empty_tape] * 3
        res = dev.batch_execute(tapes)

        assert len(res) == 3
        assert np.allclose(res[0], dev.execute(empty_tape), rtol=tol, atol=0)


class TestShotList:
    """Tests for passing shots as a list"""

    # TODO: Add tests for expval and sample with shot lists after observables are added

    def test_invalid_shot_list(self):
        """Test exception raised if the shot list is the wrong type"""
        with pytest.raises(qml.DeviceError, match="Shots must be"):
            qml.device("default.qubit", wires=2, shots=0.5)

        with pytest.raises(ValueError, match="Unknown shot sequence"):
            qml.device("default.qubit", wires=2, shots=["a", "b", "c"])

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 9), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 9),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 9), 30],
        [[(10, 3)], [(10, 3)], (3, 9), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_probs(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test a probability return"""
        dev = qml.device("default.qutrit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(U):
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(U, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        res = circuit(pnp.eye(9))

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        # TODO: Add after differentiability of qutrit circuits is implemented
        # res = qml.jacobian(circuit, argnum=[0, 1])(pnp.eye(9))

    shot_data = [
        [[1, 2, 3, 10], [(1, 1), (2, 1), (3, 1), (10, 1)], (4, 3, 2), 16],
        [
            [1, 2, 2, 2, 10, 1, 1, 5, 1, 1, 1],
            [(1, 1), (2, 3), (10, 1), (1, 2), (5, 1), (1, 3)],
            (11, 3, 2),
            27,
        ],
        [[10, 10, 10], [(10, 3)], (3, 3, 2), 30],
        [[(10, 3)], [(10, 3)], (3, 3, 2), 30],
    ]

    @pytest.mark.autograd
    @pytest.mark.parametrize("shot_list,shot_vector,expected_shape,total_shots", shot_data)
    def test_multiple_probs(self, shot_list, shot_vector, expected_shape, total_shots):
        """Test multiple probability returns"""
        dev = qml.device("default.qutrit", wires=2, shots=shot_list)

        @qml.qnode(dev)
        def circuit(U):
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(np.eye(3), wires=0)
            qml.QutritUnitary(U, wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        res = circuit(pnp.eye(9))

        assert res.shape == expected_shape
        assert circuit.device._shot_vector == shot_vector
        assert circuit.device.shots == total_shots

        # test gradient works
        # TODO: Add after differentiability of qutrit circuits is implemented
        # res = qml.jacobian(circuit, argnum=[0, 1])(pnp.eye(9))

