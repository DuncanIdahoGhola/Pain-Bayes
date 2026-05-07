# CALIBRATION SCRIPT — DS5 Digitimer / NI USB-6001
# Sends a single pulse train at a fixed intensity with NO noise (none condition).
# Use this to calibrate stimuli intensity before the main experiment.
# Plug the target intensity directly via CALIBRATION_INTENSITY_MA below.

#############################################################################
# BEGIN EXPERIMENT CODE

# Import
from scipy.signal import iirnotch, filtfilt, correlate
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Fixed Parameters ---
MAINS_FREQ_HZ = 60
MONITOR_MA_PER_V = 10.0  # DS5 monitor output: 1V = 10 mA
SAMPLE_RATE = 5000        # Hz (max compatible with NI USB-6001)

# Try to import nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
except ImportError as exc:
    raise RuntimeError(
        "DAQ is not available: nidaqmx is not installed or NI-DAQmx drivers are missing."
    ) from exc

print("\n=== Initializing Hardware ===")
try:
    system = nidaqmx.system.System.local()
    devices = [d.name for d in system.devices]
    if not devices:
        raise RuntimeError("No NI DAQ devices found. Check USB connection and NI MAX.")
    DAQ_NAME = devices[0]
    print(f"Found device: '{DAQ_NAME}'")
except Exception as exc:
    raise RuntimeError(
        "DAQ is not available: NI-DAQmx is installed but no usable device was detected."
    ) from exc


###############################################################################
# *** CALIBRATION PARAMETERS — EDIT THESE ***
###############################################################################

# --- Hardware Scaling (must match DS5 dial setting) ---
# Set to 1.0 if DS5 dial is at +/- 10 mA
# Set to 0.4 if DS5 dial is at +/- 25 mA
# Set to 0.2 if DS5 dial is at +/- 50 mA
V_PER_MA = 0.4

# --- Target Intensity ---
# This is the ONLY value you need to change between calibration trials.
# PsychoPy Builder: bind this variable to your staircase/loop output.
# Units: milliamps (mA). Starting value recommended: 1.0 mA.
CALIBRATION_INTENSITY_MA = 1.0  # <-- SET YOUR INTENSITY HERE (mA)

# --- Waveform Parameters ---
PULSE_WIDTH_MS = 2          # Width of each square pulse (ms)
PULSE_FREQUENCY_HZ = 100    # Pulse repetition rate within the train (Hz)
TRAIN_DURATION_MS = 500     # Total train duration (ms)
ZERO_PADDING_DURATION_MS = 50  # Silent buffer before and after train (ms)

# --- Output ---
# Folder where waveform CSVs and plots will be saved.
output_path = "data/calibration/"

###############################################################################


# --- Safety Check ---
max_voltage = CALIBRATION_INTENSITY_MA * V_PER_MA
if max_voltage > 10.0:
    raise ValueError(
        f"CALIBRATION_INTENSITY_MA={CALIBRATION_INTENSITY_MA} mA × V_PER_MA={V_PER_MA} "
        f"= {max_voltage:.2f} V, which exceeds the DAQ +/-10 V hardware limit. "
        "Lower CALIBRATION_INTENSITY_MA or adjust V_PER_MA."
    )


# --- Helper Functions (plotting / recording only) ---

def _notch_filter(signal, sample_rate, freq=MAINS_FREQ_HZ, quality=10.0):
    b, a = iirnotch(freq, quality, sample_rate)
    return filtfilt(b, a, signal)


def _baseline_correct(recorded, sample_rate=SAMPLE_RATE, pad_ms=ZERO_PADDING_DURATION_MS):
    n_pad = int(sample_rate * pad_ms / 1000.0)
    baseline = np.mean(recorded[:n_pad])
    return recorded - baseline


def _align_recorded(planned, recorded):
    corr = correlate(recorded, planned, mode="full")
    lag = np.argmax(corr) - (len(planned) - 1)
    if lag > 0:
        aligned = np.concatenate([recorded[lag:], np.zeros(lag)])
    elif lag < 0:
        aligned = np.concatenate([np.zeros(-lag), recorded[:lag]])
    else:
        aligned = recorded
    return aligned


# --- Waveform Generator (none condition only) ---

def generate_calibration_stimulus(
    intensity_ma,
    pulse_width_ms=PULSE_WIDTH_MS,
    pulse_frequency_hz=PULSE_FREQUENCY_HZ,
    train_duration_ms=TRAIN_DURATION_MS,
    sample_rate=SAMPLE_RATE,
):
    """
    Generates a flat (zero-noise) pulse train at exactly `intensity_ma` mA.
    All pulses in the train are identical — no noise, no variance.

    Parameters
    ----------
    intensity_ma : float
        Desired stimulation intensity in milliamps. This is the single value
        you change between calibration trials.

    Returns
    -------
    waveform_v : np.ndarray
        DAQ-ready voltage waveform (V).
    waveform_ma : np.ndarray
        Same waveform expressed in mA (for logging / plotting).
    """
    samples_per_pulse = int(round(sample_rate * pulse_width_ms / 1000.0))
    samples_per_period = int(round(sample_rate / pulse_frequency_hz))
    n_pulses = int(round(train_duration_ms * pulse_frequency_hz / 1000.0))

    # All pulses at the same flat intensity (noise_condition = "none")
    pulse_intensities = np.full(n_pulses, intensity_ma, dtype=np.float64)

    # Build pulse train: intensity held for pulse width, then zero gap
    train_2d = np.zeros((n_pulses, samples_per_period), dtype=np.float64)
    train_2d[:, :samples_per_pulse] = pulse_intensities[:, np.newaxis]
    train = train_2d.ravel()

    # Add zero padding before and after
    num_padding_samples = int(sample_rate * ZERO_PADDING_DURATION_MS / 1000.0)
    padding = np.zeros(num_padding_samples)
    waveform_ma = np.concatenate((padding, train, padding)).astype(np.float64)

    # Convert to voltage for DAQ
    waveform_v = waveform_ma * V_PER_MA

    # Hardware safety check
    max_v = np.max(np.abs(waveform_v))
    if max_v > 10.0:
        raise ValueError(
            f"Waveform peak of {max_v:.2f} V exceeds DAQ +/-10 V limit!"
        )

    return waveform_v, waveform_ma


# --- DAQ Fire + Record ---

def fire_and_record(waveform_v, output_path, trial_number, intensity_ma, DAQ_NAME=DAQ_NAME):
    """
    Sends the waveform via ao0 and simultaneously records on ai0.
    Saves a CSV and a plot to output_path.
    """
    total_samples = len(waveform_v)

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:

        ao_task.ao_channels.add_ao_voltage_chan(f"{DAQ_NAME}/ao0")
        ai_task.ai_channels.add_ai_voltage_chan(
            f"{DAQ_NAME}/ai0", terminal_config=TerminalConfiguration.RSE
        )

        ao_task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples,
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            source="OnboardClock",
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples,
        )

        ao_task.write(waveform_v, auto_start=False)
        ai_task.start()
        ao_task.start()
        ao_task.wait_until_done(timeout=1.0)

        trial_recording_v = ai_task.read(
            number_of_samples_per_channel=total_samples, timeout=2.0
        )
        trial_recording_v = _notch_filter(np.array(trial_recording_v), SAMPLE_RATE)

        ao_task.stop()
        ai_task.stop()

    planned_ma = np.array(waveform_v) / V_PER_MA
    recorded_ma = _baseline_correct(np.array(trial_recording_v) * MONITOR_MA_PER_V)
    aligned_recorded_ma = _align_recorded(planned_ma, recorded_ma)

    os.makedirs(output_path, exist_ok=True)
    base_name = f"calib_trial_{trial_number}_{intensity_ma:.2f}mA"
    csv_path = os.path.join(output_path, f"{base_name}.csv")
    png_path = os.path.join(output_path, f"{base_name}.png")

    time_ms = np.arange(total_samples) / SAMPLE_RATE * 1000
    csv_data = np.column_stack([time_ms, planned_ma, aligned_recorded_ma])
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header="time_ms,planned_ma,recorded_aligned_ma",
        comments="",
    )
    print(f"Saved: {csv_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(time_ms, planned_ma, color="steelblue", linewidth=0.8,
            label="Planned (mA)", alpha=0.8)
    ax.plot(time_ms, aligned_recorded_ma, color="tomato", linewidth=0.8,
            label="Recorded aligned (mA)", alpha=0.9)
    r_val = np.corrcoef(planned_ma, aligned_recorded_ma)[0, 1]
    ax.set_title(f"Calibration trial {trial_number} — {intensity_ma:.2f} mA  |  r = {r_val:.3f}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (mA)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


#############################################################################
# TRIAL LOOP — put this block inside your PsychoPy trial routine
#############################################################################

# --- Variables to retrieve from PsychoPy Builder ---
# CALIBRATION_INTENSITY_MA : float  ← bind to your staircase / loop variable
#                                      (e.g. thisCalib['intensity'] or a slider)
#                                      Starts at 1.0 mA and changes by your chosen increment.
# trial_number             : int    ← e.g. calibration_loop.thisN

# Example (replace with your actual PsychoPy variable names):
# CALIBRATION_INTENSITY_MA = thisCalib['intensity']   # mA — from your loop/staircase
# trial_number = calibration_loop.thisN



class DS5PulseConfig:
    """Configuration object for the DS5 stimulator."""
    def __init__(
        self,
        v_per_ma=0.4,
        pulse_width_ms=2,
        pulse_frequency_hz=100,
        train_duration_ms=500,
        record_monitor=True,
    ):
        self.v_per_ma = v_per_ma
        self.pulse_width_ms = pulse_width_ms
        self.pulse_frequency_hz = pulse_frequency_hz
        self.train_duration_ms = train_duration_ms
        self.record_monitor = record_monitor


def fire_none_uncertainty_pulse(intensity_ma, config, output_dir, trial_label):
    """
    Generate and fire a flat (no-noise) pulse at intensity_ma mA.
    Called once per trial by the PsychoPy experiment.
    """
    global V_PER_MA, PULSE_WIDTH_MS, PULSE_FREQUENCY_HZ, TRAIN_DURATION_MS

    # Apply config values
    V_PER_MA = config.v_per_ma
    PULSE_WIDTH_MS = config.pulse_width_ms
    PULSE_FREQUENCY_HZ = config.pulse_frequency_hz
    TRAIN_DURATION_MS = config.train_duration_ms

    waveform_v, waveform_ma = generate_calibration_stimulus(
        intensity_ma=intensity_ma,
        pulse_width_ms=config.pulse_width_ms,
        pulse_frequency_hz=config.pulse_frequency_hz,
        train_duration_ms=config.train_duration_ms,
    )

    fire_and_record(
        waveform_v=waveform_v,
        output_path=str(output_dir),
        trial_number=trial_label,
        intensity_ma=intensity_ma,
    )

    return {"intensity_ma": intensity_ma, "label": trial_label}