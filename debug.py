import pennylane as qml
from pennylane.capture.capture_meta import meas_counts, op_counts

print(op_counts)
print(meas_counts)

qml.PauliX(0)
qml.state()

print(op_counts)
print(meas_counts)
