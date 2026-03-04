import pandas as pd
import numpy as np
import os
import itertools

# --- Configuration ---
n_trials = 192
num_files = 100
output_folder = "condition_files"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Factor levels ---
expected_vas_levels = [30, 70]
expect_uncertainty_levels = ['low', 'high']
sensory_uncertainty_levels = ['low', 'high', 'none']
stim_intensity_levels = [30, 70]

# --- Build FULL factorial design (24 conditions) ---
factorial_conditions = list(itertools.product(
    expected_vas_levels,
    expect_uncertainty_levels,
    sensory_uncertainty_levels,
    stim_intensity_levels
))

assert len(factorial_conditions) == 24

# --- Repeat each condition 8 times (192 trials total) ---
repetitions = 8
all_conditions = factorial_conditions * repetitions

for file_idx in range(1, num_files + 1):

    df = pd.DataFrame(all_conditions, columns=[
        'expected_vas',
        'expect_uncertainty',
        'sensory_uncertainty',
        'stim_intensity'
    ])

    # Shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)

    # --- Create independent check_up column ---
    n_yes = 32
    n_no = n_trials - n_yes  # 160

    check_values = ['yes'] * n_yes + ['no'] * n_no
    np.random.shuffle(check_values)

    df['check_up'] = check_values


    # Save file
    filename = os.path.join(output_folder, f"{file_idx:03d}_conditions.csv")
    df.to_csv(filename, index=False)

    print(f"Generated {filename}")

print(f"\nAll {num_files} condition files generated in the '{output_folder}' folder.")