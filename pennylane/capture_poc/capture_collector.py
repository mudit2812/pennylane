import os
import json
import datetime
import atexit
from collections import Counter # <--- IMPORT COUNTER HERE

# Define the global Counters *within this module*
# These will be the canonical source of truth for counts
_op_counts = Counter()
_meas_counts = Counter()

# --- Configuration ---
TELEMETRY_DIR = os.path.join(os.path.expanduser("~"), ".pennylane_poc_telemetry")
TELEMETRY_FILE = os.path.join(TELEMETRY_DIR, "usage_data.json")

_telemetry_enabled = False # Global flag for telemetry status

# --- Core Functions ---
def _load_counters_from_disk():
    """Loads historical aggregated counts from a local file."""
    global _op_counts, _meas_counts # <--- IMPORTANT: Declare as global to modify
    
    if not os.path.exists(TELEMETRY_FILE):
        return

    try:
        with open(TELEMETRY_FILE, 'r') as f:
            data = json.load(f)
            # Update *our* internal Counters
            _op_counts.update(data.get("op_counts", {}))
            _meas_counts.update(data.get("meas_counts", {}))
        print(f"PennyLane Capture POC: Loaded existing data from {TELEMETRY_FILE}")
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"PennyLane Capture POC: Error loading {TELEMETRY_FILE}: {e}. Starting fresh.")
        # If there's an error, we keep _op_counts and _meas_counts as whatever they currently are (likely empty)


def _save_counters_to_disk():
    """Saves current global counts to the local file."""
    if not _telemetry_enabled:
        print("PennyLane Capture POC: Not saving data because capture is disabled.")
        return

    os.makedirs(TELEMETRY_DIR, exist_ok=True)

    try:
        # Convert the Counter objects to dictionaries with string keys
        op_counts_str_keys = {cls: count for cls, count in _op_counts.items()} # Use _op_counts
        meas_counts_str_keys = {cls: count for cls, count in _meas_counts.items()} # Use _meas_counts

        data = {
            "last_updated": datetime.datetime.now().isoformat(),
            "op_counts": op_counts_str_keys,
            "meas_counts": meas_counts_str_keys
        }

        with open(TELEMETRY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"PennyLane Capture POC: Data saved to {TELEMETRY_FILE}")
    except Exception as e:
        print(f"PennyLane Capture POC: Error saving data to {TELEMETRY_FILE}: {e}")

def initialize_capture_poc():
    """Initializes the capture system, checks opt-out, and sets up save hooks."""
    global _telemetry_enabled # <--- Declare as global to modify

    if os.environ.get("PENNYLANETELEMETRY", "").lower() == "on":
        _telemetry_enabled = True
        print("PennyLane Capture POC: Enabled. Set PENNYLANETELEMETRY=off to disable.")

        _load_counters_from_disk() # Load existing data here

        atexit.register(_save_counters_to_disk)
        return

    print("PennyLane Capture POC: Opted out via PENNYLANETELEMETRY=off.")

# --- Expose public interface for other modules to increment ---
def increment_operator_count(op_class_name):
    """Increment a counter for a given operator class name."""
    if _telemetry_enabled:
        _op_counts[op_class_name] += 1

def increment_measurement_count(meas_class_name):
    """Increment a counter for a given measurement class name."""
    if _telemetry_enabled:
        _meas_counts[meas_class_name] += 1

# --- Getter for current counts (for debug.py and view_telemetry_data.py) ---
def get_op_counts():
    return _op_counts

def get_meas_counts():
    return _meas_counts

# --- Automatic Initialization ---
initialize_capture_poc()