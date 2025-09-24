import os
import pennylane as qml
# --- IMPORTANT CHANGE: Import from the collector ---
from pennylane.capture_poc.capture_collector import get_op_counts, get_meas_counts, TELEMETRY_FILE
# ---------------------------------------------------

print("\n--- PennyLane PoC Telemetry Demo ---")
print(f"Local telemetry data file: {TELEMETRY_FILE}")

# Initial state (should reflect loaded data if file existed and telemetry is enabled)
print("\nInitial Counts (in-memory):")
print(f"  Operations: {get_op_counts()}") # Call getter function
print(f"  Measurements: {get_meas_counts()}") # Call getter function

# --- User's PennyLane Code (example) ---
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def my_circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.5, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

print("\n>>> Executing my_circuit 3 times <<<")
my_circuit(0.1)
my_circuit(0.2)
my_circuit(0.3)

@qml.qnode(dev)
def another_circuit():
    qml.Hadamard(0)
    qml.PauliX(1)
    return qml.probs(wires=0)

# Final state before script exits
print("\nFinal Counts (in-memory, before saving):")
print(f"  Operations: {get_op_counts()}") # Call getter function
print(f"  Measurements: {get_meas_counts()}") # Call getter function

print("\n--- Demo Complete ---")
print(f"Telemetry data will be saved to '{TELEMETRY_FILE}' upon script exit (if enabled).")
print("To inspect the saved data, check the file directly or use the 'view_telemetry_data.py' script (coming Wednesday).")
