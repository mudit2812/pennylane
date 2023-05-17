# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains tools and decorators for registering qfunc transforms."""
# pylint: disable=too-few-public-methods
from copy import deepcopy
import functools
import inspect
import os
import warnings

import pennylane as qml
from pennylane.tape import make_qscript


def make_tape(fn):
    """Returns a function that generates the tape from a quantum function without any
    operation queuing taking place.

    This is useful when you would like to manipulate or transform
    the tape created by a quantum function without evaluating it.

    Args:
        fn (function): the quantum function to generate the tape from

    Returns:
        function: The returned function takes the same arguments as the quantum
        function. When called, it returns the generated quantum tape
        without any queueing occuring.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)

    We can use ``make_tape`` to extract the tape generated by this
    quantum function, without any of the operations being queued by
    any existing queuing contexts:

    >>> with qml.tape.QuantumTape() as active_tape:
    ...     qml.RY(1.0, wires=0)
    ...     tape = make_tape(qfunc)(0.5)
    >>> tape.operations
    [Hadamard(wires=[0]), CNOT(wires=[0, 1]), RX(0.5, wires=[0])]

    Note that the currently recording tape did not queue any of these quantum operations:

    >>> active_tape.operations
    [RY(1.0, wires=[0])]
    """

    def wrapper(*args, **kwargs):
        with qml.QueuingManager.stop_recording(), qml.tape.QuantumTape() as new_tape:
            fn(*args, **kwargs)

        return new_tape

    return wrapper


class single_tape_transform:
    """For registering a tape transform that takes a tape and outputs a single new tape.

    Examples of such transforms include circuit compilation.

    Args:
        transform_fn (function): The function to register as the single tape transform.
            It can have an arbitrary number of arguments, but the first argument
            **must** be the input tape.

    **Example**

    A valid single tape transform is a quantum function that satisfies the following:

    - The first argument must be an input tape

    - Depending on the structure of this input tape, various quantum operations, functions,
      and templates may be called.

    - Any internal classical processing should use the ``qml.math`` module to ensure
      the transform is differentiable.

    - There is no return statement.

    For example:

    .. code-block:: python

        @qml.single_tape_transform
        def my_transform(tape, x, y):
            # loop through all operations on the input tape
            for op in tape:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]

                    qml.RX(x * qml.math.abs(param), wires=wires[1])
                    qml.RY(y * qml.math.abs(param), wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

    This transform iterates through the input tape, and replaces any :class:`~.CRX` operation with
    two single qubit rotations and a :class:`~.CZ` operation. These newly queued operations will
    form the output transformed tape.

    We can apply this transform to a quantum tape:

    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.Hadamard(wires=0)
    ...     qml.CRX(-0.5, wires=[0, 1])
    >>> new_tape = my_transform(tape, 1., 2.)
    >>> print(qml.drawer.tape_text(new_tape, decimals=1))
    0: ──H────────────────╭Z─┤
    1: ──RX(0.5)──RY(1.0)─╰●─┤

    """

    def __init__(self, transform_fn):
        if not callable(transform_fn):
            raise ValueError(
                f"The tape transform function to register, {transform_fn}, "
                "does not appear to be a valid Python function or callable."
            )

        self.transform_fn = transform_fn
        functools.update_wrapper(self, transform_fn)

    def __call__(self, tape, *args, **kwargs):
        with qml.queuing.AnnotatedQueue() as q:
            self.transform_fn(tape, *args, **kwargs)
        qs = qml.tape.QuantumScript.from_queue(q, shots=tape.shots)
        for obj, info in q.items():
            qml.queuing.QueuingManager.append(obj, **info)
        return qs


def _create_qfunc_internal_wrapper(fn, tape_transform, transform_args, transform_kwargs):
    """Convenience function to create the internal wrapper function
    generated by the qfunc_transform decorator"""
    if isinstance(fn, qml.Device):
        new_dev = deepcopy(fn)

        @new_dev.custom_expand
        def new_expand_fn(self, tape, *args, **kwargs):  # pylint: disable=unused-variable
            tape = tape_transform(tape, *transform_args, **transform_kwargs)
            return self.default_expand_fn(tape, *args, **kwargs)

        return new_dev

    if isinstance(fn, qml.tape.QuantumScript):
        return tape_transform(fn, *transform_args, **transform_kwargs)

    if not callable(fn):
        raise ValueError(
            f"The qfunc to transform, {fn}, does not appear "
            "to be a valid Python function or callable."
        )
    if isinstance(fn, qml.QNode):
        raise ValueError("QNodes cannot be declared as qfunc transforms.")

    @functools.wraps(fn)
    def internal_wrapper(*args, **kwargs):
        tape = make_qscript(fn)(*args, **kwargs)
        tape = tape_transform(tape, *transform_args, **transform_kwargs)

        num_measurements = len(tape.measurements)
        if num_measurements == 0:
            return None
        return tape.measurements[0] if num_measurements == 1 else tape.measurements

    return internal_wrapper


def qfunc_transform(tape_transform):
    """Given a function which defines a tape transform, convert the function into
    one that applies the tape transform to quantum functions (qfuncs).

    Args:
        tape_transform (function or single_tape_transform): the single tape transform
            to convert into the qfunc transform.

    Returns:
        function: A qfunc transform, that acts on any qfunc, and returns a *new*
        qfunc as per the tape transform. Note that if ``tape_transform`` takes
        additional parameters beyond a single tape, then the created qfunc transform
        will take the *same* parameters, prior to being applied to the qfunc.

    **Example**

    Given a single tape transform ``my_transform(tape, x, y)``, you can use
    this function to convert it into a qfunc transform:

    >>> my_qfunc_transform = qfunc_transform(my_transform)

    It can then be used to transform an existing qfunc:

    >>> new_qfunc = my_qfunc_transform(0.6, 0.7)(old_qfunc)
    >>> new_qfunc(params)

    It can also be used as a decorator:

    .. code-block:: python

        @qml.qfunc_transform
        def my_transform(tape, x, y):
            for op in tape:
                if op.name == "CRX":
                    wires = op.wires
                    param = op.parameters[0]
                    qml.RX(x * param, wires=wires[1])
                    qml.RY(y * qml.math.sqrt(param), wires=wires[1])
                    qml.CZ(wires=[wires[1], wires[0]])
                else:
                    op.queue()

        @my_transform(0.6, 0.1)
        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

    >>> dev = qml.device("default.qubit", wires=2)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(2.5))
    0: ──H──────────────────╭Z─┤
    1: ──RX(1.50)──RY(0.16)─╰●─┤  <Z>

    The transform weights provided to a qfunc transform are fully differentiable,
    allowing the transform itself to be differentiated and trained. For more details,
    see the Differentiability section under Usage Details.

    .. details::
        :title: Usage Details

        **Inline usage**

        qfunc transforms, when used inline (that is, not as a decorator), take the following form:

        >>> my_transform(transform_weights)(ansatz)(param)

        or

        >>> my_transform(ansatz)(param)

        if they do not permit any parameters. We can break this down into distinct steps,
        to show what is happening with each new function call:

        0. Create a transform defined by the transform weights:

           >>> specific_transform = my_transform(transform_weights)

           Note that this step is skipped if the transform does not provide any
           weights/parameters that can be modified!

        1. Apply the transform to the qfunc. A qfunc transform always acts on
           a qfunc, returning a new qfunc:

           >>> new_qfunc = specific_transform(ansatz)

        2. Finally, we evaluate the new, transformed, qfunc:

           >>> new_qfunc(params)

        So the syntax

        >>> my_transform(transform_weights)(ansatz)(param)

        simply 'chains' these three steps together, into a single call.

        **Differentiability**

        When applying a qfunc transform, not only is the newly transformed qfunc fully
        differentiable, but the qfunc transform parameters *themselves* are differentiable.
        This allows us to train both the quantum function, as well as the transform
        that created it.

        Consider the following example, where a pre-defined ansatz is transformed
        within a QNode:

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            def ansatz(x):
                qml.Hadamard(wires=0)
                qml.CRX(x, wires=[0, 1])

            @qml.qnode(dev)
            def circuit(param, transform_weights):
                qml.RX(0.1, wires=0)

                # apply the transform to the ansatz
                my_transform(*transform_weights)(ansatz)(param)

                return qml.expval(qml.PauliZ(1))

        We can print this QNode to show that the qfunc transform is taking place:

        >>> x = np.array(0.5, requires_grad=True)
        >>> y = np.array([0.1, 0.2], requires_grad=True)
        >>> print(qml.draw(circuit)(x, y))
        0: ──RX(0.10)──H────────╭Z─┤
        1: ──RX(0.05)──RY(0.14)─╰●─┤  <Z>

        Evaluating the QNode, as well as the derivative, with respect to the gate
        parameter *and* the transform weights:

        >>> circuit(x, y)
        0.9887793925354269
        >>> qml.grad(circuit)(x, y)
        (array(-0.02485651), array([-0.02474011, -0.09954244]))

        **Implementation details**

        Internally, the qfunc transform works as follows:

        .. code-block:: python

            def transform(old_qfunc, params):
                def new_qfunc(*args, **kwargs):
                    # 1. extract the QuantumTape from the old qfunc, being
                    # careful *not* to have it queued.
                    tape = make_qscript(old_qfunc)(*args, **kwargs)

                    # 2. transform the tape
                    new_tape = tape_transform(tape, params)

                    # 3. queue the *new* tape to the active queuing context
                    new_tape.queue()
                return new_qfunc

        *Note: this is pseudocode; the actual implementation is significantly more complicated!*

        Steps (1) and (3) are identical for all qfunc transforms; it is only step (2),
        ``tape_transform`` and the corresponding tape transform parameters, that define the qfunc
        transformation.

        That is, given a tape transform that **defines the qfunc transformation**, the
        decorator **elevates** the tape transform to one that works on quantum functions
        rather than tapes. This decorator therefore automates the process of adding in
        the queueing logic required under steps (1) and (3), so that it does not need to be
        repeated and tested for every new qfunc transform.
    """
    if os.environ.get("SPHINX_BUILD") == "1":
        # If called during a Sphinx documentation build,
        # simply return the original function rather than
        # instantiating the object. This allows the signature to
        # be correctly displayed in the documentation.

        warnings.warn(
            "qfunc transformations have been disabled, as a Sphinx "
            "build has been detected via SPHINX_BUILD='1'. If this is not the "
            "case, please set the environment variable SPHINX_BUILD='0'.",
            UserWarning,
        )

        return tape_transform

    if not callable(tape_transform):
        raise ValueError(
            "The qfunc_transform decorator can only be applied "
            "to single tape transform functions."
        )

    if not isinstance(tape_transform, single_tape_transform):
        tape_transform = single_tape_transform(tape_transform)

    sig = inspect.signature(tape_transform)
    params = sig.parameters

    if len(params) > 1:

        @functools.wraps(tape_transform)
        def make_qfunc_transform(*targs, **tkwargs):
            def wrapper(fn):
                return _create_qfunc_internal_wrapper(fn, tape_transform, targs, tkwargs)

            wrapper.tape_fn = functools.partial(tape_transform, *targs, **tkwargs)

            return wrapper

    elif len(params) == 1:

        @functools.wraps(tape_transform)
        def make_qfunc_transform(fn):
            return _create_qfunc_internal_wrapper(fn, tape_transform, [], {})

    make_qfunc_transform.tape_fn = tape_transform
    return make_qfunc_transform
