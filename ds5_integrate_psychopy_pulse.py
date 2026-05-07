# METHODS DESCRIPTION
# Electrical pain stimuli were delivered continuously via an isolated bipolar constant current stimulator (Model DS5, Digitimer Ltd.), driven by a National Instruments data acquisition interface (NI USB-6001).

#  To isolate the perceptual effect of stimulus variance (noise) from absolute peak intensity, the stimulation waveforms were constructed using a predefined mixture model. This approach strictly controls for the "peak-picker" confound—grounded in the peak-end rule (Kahneman et al., 1993)—where participants may disproportionately weight the maximum intensity of an experience. Without this control, highly variable stimuli might be rated as more painful simply because they randomly hit severe pain thresholds more frequently. The the noise waveform was generated at an effective frequency of 5 kHz (1000 samples form the 200 ms duration) and the DAQ hardware output rate was maintained at 5 kHz.
#
# For each trial, a baseline current ($I_{base}$) and a maximum noise half-width ($a_{max}$) were calculated strictly within the participant's individually calibrated pain threshold and pain tolerance, and the effective half-width was set as $a = NOISE_WIDTH_PROPORTION \times a_{max}$. To manipulate variance while holding both the absolute peak intensity and the frequency of those peaks constant, a spike-injection method was utilized. Within the 200 ms active pulse window, exactly 5% of the effective noise steps were explicitly set to the absolute maximum bound ($I_{base} + a$) and another 5% were set to the absolute minimum bound ($I_{base} - a$). Variance was exclusively manipulated via the remaining 90% of the stimulus samples. In the low-noise condition, this background consisted of normally distributed noise tightly clustered around the mean. In the high-noise condition, the background consisted of uniformly distributed noise spanning the entire range between the bounds. All steps were subsequently randomly shuffled in time. Consequently, both conditions contained the exact same number of extreme physical stimulus events, isolating the overall signal variance as the sole manipulated variable.variable.

# TEST REMOVE TO INTEGRATE IN PSYCHOPY
expInfo = dict()
expInfo["threshold"] = str(5)
expInfo["tolerance"] = str(9.0)


#############################################################################
# À PARTIR D'ICI, METTRE DANS LE DÉBUT DE L'EXPÉRIENCE (BEGIN EXPERIMENT)

# BEGIN EXPERIMENT CODE

# Import
from scipy.stats import gaussian_kde
from scipy.signal import iirnotch, filtfilt, correlate
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Fixed Parameters ---
# Power line frequency to filter for plots. Should always be 60
MAINS_FREQ_HZ = 60
# Scaling in the monitoring port. Should always be 10.
MONITOR_MA_PER_V = 10.0  # DS5 monitor output: 1V = 10 mA
SAMPLE_RATE = 5000  # Hz (Max ompatible with NI USB-6001)


# Try to import nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
except ImportError as exc:
    DAQ_AVAILABLE = False
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
    if len(devices) > 1:
        print(f"Multiple devices found {devices}, using '{DAQ_NAME}'.")
    else:
        print(f"Found device: '{DAQ_NAME}'")
    DAQ_AVAILABLE = True
except Exception as exc:
    DAQ_AVAILABLE = False
    raise RuntimeError(
        "DAQ is not available: NI-DAQmx is installed but no usable device was detected."
    ) from exc


###############################################################
# --- Parameters that could change after piloting ---
###############################################################

# --- Hardware Scaling (Digitimer DS5 Settings) ---
# Set to 1.0 if DS5 dial is at +/- 10 mA (10V from DAQ = 10mA to subject)
# Set to 0.4 if DS5 dial is at +/- 25 mA (10V from DAQ = 25mA to subject)
# Set to 0.2 if DS5 dial is at +/- 50 mA (10V from DAQ = 50mA to subject)
V_PER_MA = 0.4

# Waveform Parameters ---
# Stimulation is delivered as a train of brief square pulses.
PULSE_WIDTH_MS = 2  # width of each pulse within the train
PULSE_FREQUENCY_HZ = 100  # pulse repetition rate within the train
TRAIN_DURATION_MS = 500  # total duration of the pulse train
ZERO_PADDING_DURATION_MS = 50


# Design Parameters ---
PAIN_PROPORTIONS = {"low": 0.25, "high": 0.75}
# Fraction of the maximum allowed symmetric half-width used for noise spikes.
# In compute_base_and_width():
#   a_max = min(I_base - pain_threshold, pain_tolerance - I_base)
#   a = NOISE_WIDTH_PROPORTION * a_max
# Using 0.8 means noise range reaches 80% of bounds around I_base (within threshold/tolerance).
NOISE_WIDTH_PROPORTION = 0.8
EXTREME_FRACTION = 0.10  # 10% of pulses set to exact MAX, 10% set to exact MIN


# --- Participant-Specific Parameters (Calibrated per participant) ---
PAIN_THRESHOLD = float(expInfo["threshold"])  # in mA
PAIN_TOLERANCE = float(expInfo["tolerance"])

# Sanity check
assert (
    PAIN_THRESHOLD < PAIN_TOLERANCE
), "Pain threshold must be less than pain tolerance!"
# Make sure tolerance is at least 2 mA above threshold to allow for meaningful noise manipulation
assert (
    PAIN_TOLERANCE - PAIN_THRESHOLD >= 2.0
), "Pain tolerance must be at least 2 mA above threshold for effective noise manipulation!"
# MAke sure scaling is not set such that we exceed DAQ limits
max_possible_ma = PAIN_TOLERANCE + (
    NOISE_WIDTH_PROPORTION * (PAIN_TOLERANCE - PAIN_THRESHOLD) * 0.8
)  # Max base + max noise
max_possible_v = max_possible_ma * V_PER_MA
if max_possible_v > 10.0:
    raise ValueError(
        f"With current settings, max possible voltage to subject is {max_possible_v:.2f}V, which exceeds DAQ limits. Adjust V_PER_MA or noise parameters."
    )


# Plotting functions


def _notch_filter(signal, sample_rate, freq=MAINS_FREQ_HZ, quality=10.0):
    """Applies a notch filter to remove mains interference.
    only used for plotting."""
    b, a = iirnotch(freq, quality, sample_rate)
    return filtfilt(b, a, signal)


def _baseline_correct(
    recorded, sample_rate=SAMPLE_RATE, pad_ms=ZERO_PADDING_DURATION_MS
):
    """Subtracts the mean of the leading zero-padding region (known 0 mA period).
    only used for plotting."""
    n_pad = int(sample_rate * pad_ms / 1000.0)
    baseline = np.mean(recorded[:n_pad])
    return recorded - baseline


def _align_recorded(planned, recorded):
    """Shifts recorded to align with planned via cross-correlation, zero-pads the tail.
    only used for plotting."""
    corr = correlate(recorded, planned, mode="full")
    lag = np.argmax(corr) - (len(planned) - 1)
    if lag > 0:
        aligned = np.concatenate([recorded[lag:], np.zeros(lag)])
    elif lag < 0:
        aligned = np.concatenate([np.zeros(-lag), recorded[:lag]])
    else:
        aligned = recorded
    return aligned


def compute_base_and_width(pain_threshold, pain_tolerance, intensity_condition):
    """Calculates baseline current and max allowable noise half-width."""
    if intensity_condition not in PAIN_PROPORTIONS:
        raise ValueError(
            f"Invalid intensity_condition '{intensity_condition}'. "
            "Expected one of: low, high."
        )
    pain_prop = PAIN_PROPORTIONS[intensity_condition]
    painful_range = pain_tolerance - pain_threshold
    I_base = pain_threshold + pain_prop * painful_range
    a_max = min(I_base - pain_threshold, pain_tolerance - I_base)
    a = NOISE_WIDTH_PROPORTION * a_max
    return I_base, a


def generate_controlled_peak_stimulus(
    intensity_condition,
    noise_condition,
    pain_threshold=PAIN_THRESHOLD,
    pain_tolerance=PAIN_TOLERANCE,
    pulse_width_ms=PULSE_WIDTH_MS,
    pulse_frequency_hz=PULSE_FREQUENCY_HZ,
    train_duration_ms=TRAIN_DURATION_MS,
    sample_rate=SAMPLE_RATE,
):
    """
    Generates one condition-specific pulse train, converts it to DAQ voltage,
    and verifies safety limits.

    The train is a sequence of square pulses of width `pulse_width_ms`
    repeating at `pulse_frequency_hz` for a total of `train_duration_ms`.
    Variance is manipulated per-pulse: each pulse's intensity is independently
    drawn under the spike-injection scheme, so within-pulse samples are flat.
    """
    if intensity_condition not in ("low", "high"):
        raise ValueError(
            f"Invalid intensity_condition '{intensity_condition}'. Expected 'low' or 'high'."
        )
    if noise_condition not in ("low", "high", "none"):
        raise ValueError(
            f"Invalid noise_condition '{noise_condition}'. Expected 'low', 'high', or 'none'."
        )

    I_base, a = compute_base_and_width(
        pain_threshold, pain_tolerance, intensity_condition
    )

    # --- Train geometry ---
    samples_per_pulse = int(round(sample_rate * pulse_width_ms / 1000.0))
    samples_per_period = int(round(sample_rate / pulse_frequency_hz))
    samples_per_gap = samples_per_period - samples_per_pulse
    if samples_per_gap < 0:
        raise ValueError(
            f"Pulse width ({pulse_width_ms} ms) exceeds period at "
            f"{pulse_frequency_hz} Hz."
        )
    n_pulses = int(round(train_duration_ms * pulse_frequency_hz / 1000.0))

    # --- Per-pulse intensities ---
    if noise_condition == "none":
        pulse_intensities = np.full(n_pulses, I_base, dtype=np.float64)
        a_used = 0.0
    else:
        num_extremes = int(n_pulses * EXTREME_FRACTION)
        num_background = n_pulses - (2 * num_extremes)

        max_spikes = np.full(num_extremes, I_base + a)
        min_spikes = np.full(num_extremes, I_base - a)

        if noise_condition == "low":
            bg_noise = np.random.normal(I_base, a * 0.1, num_background)
        else:
            bg_noise = np.random.uniform(
                I_base - (a * 0.95), I_base + (a * 0.95), num_background
            )
        bg_noise = np.clip(bg_noise, I_base - (a * 0.99), I_base + (a * 0.99))

        pulse_intensities = np.concatenate([max_spikes, min_spikes, bg_noise])
        np.random.shuffle(pulse_intensities)
        a_used = a

    # --- Build the train: pulse_intensities[k] held for samples_per_pulse, then zero gap ---
    train_2d = np.zeros((n_pulses, samples_per_period), dtype=np.float64)
    train_2d[:, :samples_per_pulse] = pulse_intensities[:, np.newaxis]
    train = train_2d.ravel()

    # --- Add zero padding ---
    num_padding_samples = int(sample_rate * ZERO_PADDING_DURATION_MS / 1000.0)
    padding = np.zeros(num_padding_samples)
    waveform_ma = np.concatenate((padding, train, padding)).astype(np.float64)

    # Convert to Voltage for the DAQ
    waveform_v = waveform_ma * V_PER_MA

    # Hardware Safety Check (NI USB-6001 hard limit is +/- 10.0V)
    max_requested_v = np.max(np.abs(waveform_v))
    if max_requested_v > 10.0:
        raise ValueError(
            f"CRITICAL ERROR in intensity={intensity_condition}, noise={noise_condition}: "
            f"Requested {max_requested_v:.2f}V exceeds DAQ +/- 10V limit!"
        )

    analysis = (waveform_ma, train, I_base, a_used)
    return waveform_v, analysis


def fire_and_record(
    daq_vectors_v, output_path, trial_number, condition_label="trial", DAQ_NAME=DAQ_NAME
):
    """
    Runs the DAQ playback and recording loop.
    Hardware-syncs the analog input (ai0) to the analog output (ao0) trigger.
    """
    recorded_data = []
    total_samples = int(
        SAMPLE_RATE * (TRAIN_DURATION_MS + 2 * ZERO_PADDING_DURATION_MS) / 1000
    )

    # Create TWO tasks: one for sending (AO), one for receiving (AI)
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:

        # Setup Channels
        ao_task.ao_channels.add_ao_voltage_chan(f"{DAQ_NAME}/ao0")
        ai_task.ai_channels.add_ai_voltage_chan(
            f"{DAQ_NAME}/ai0", terminal_config=TerminalConfiguration.RSE
        )

        # Setup Timing (Both run at identical rates for identical durations)
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

        # Pre-load output data into the DAQ buffer
        ao_task.write(daq_vectors_v, auto_start=False)

        # Start AI first so it is already listening when AO begins
        ai_task.start()
        ao_task.start()

        # 3. Wait for the physical pulse to finish
        ao_task.wait_until_done(timeout=1.0)

        # 4. Read the recorded data from the AI buffer
        trial_recording_v = ai_task.read(
            number_of_samples_per_channel=total_samples, timeout=2.0
        )
        trial_recording_v = _notch_filter(np.array(trial_recording_v), SAMPLE_RATE)
        recorded_data.append(trial_recording_v)

        # Reset task states for the next trial
        ao_task.stop()
        ai_task.stop()

    # Convert planned voltage -> mA and recorded monitor voltage -> mA
    planned_ma = np.array(daq_vectors_v) / V_PER_MA
    recorded_ma = _baseline_correct(np.array(trial_recording_v) * MONITOR_MA_PER_V)
    aligned_recorded_ma = _align_recorded(planned_ma, recorded_ma)

    # Save aligned data and plot to output path
    os.makedirs(output_path, exist_ok=True)
    safe_label = str(condition_label).replace(" ", "_")
    base_name = f"trial_{trial_number}_{safe_label}"
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
    print(f"Recorded aligned waveform saved to {csv_path}")

    # Plot planned vs aligned recorded waveform
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(
        time_ms,
        planned_ma,
        color="steelblue",
        linewidth=0.8,
        label="Planned (mA)",
        alpha=0.8,
    )
    ax.plot(
        time_ms,
        aligned_recorded_ma,
        color="tomato",
        linewidth=0.8,
        label="Recorded aligned (mA)",
        alpha=0.9,
    )
    r_val = np.corrcoef(planned_ma, aligned_recorded_ma)[0, 1]
    ax.set_title(f"{condition_label}  |  r = {r_val:.3f}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (mA)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Recorded aligned plot saved to {png_path}")

    return recorded_data


#############################################################################
# À PARTIR D'ICI, METTRE DANS LA LOOP DE CHAQUE ESSAI


# retrieve current conditions (TO ADAPT TO THE TASK)
noise_condition = (
    sensory_uncertainty  # 'high' or 'low', determined by your trial sequence
)
intensity_condition = (
    stim_intensity  # 'high' or 'low', determined by your trial sequence
)
trial_number = trial_counter  # Retrieve from psychopy
# Define your output path (in subject data folder), retrieve from psychopy if needed.
# THE DEVICE WILL RECORD THE mA AND MAKE PLOT FOR EACH TRIAL
output_path = "data/%s/bayes_pain/uncertainty_figs/" % (expInfo["participant"])


# GENERATE WAVE FOR THIS CONDITION (CAN BE AT THE START OF THE TRIAL e.g. during cross)
trial_wave_v, trial_analysis = generate_controlled_peak_stimulus(
    noise_condition=noise_condition, intensity_condition=intensity_condition
)

# FIRE SHOCK
fire_and_record(
    trial_wave_v,
    output_path=output_path,
    trial_number=trial_number,
    condition_label=f"{intensity_condition}_pain_{noise_condition}_noise",
)
