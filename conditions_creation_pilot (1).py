import pandas as pd
import numpy as np
import os

# --- Configuration ---
output_folder = "condition_files_pilot"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Factor levels ---
sensory_uncertainty_levels = ['low', 'high', 'none']
stim_intensity_levels = ['low', 'high']

# --- Build trial list: 15 trials per (sensory_uncertainty x stim_intensity) cell ---
# 3 noise levels x 2 intensity levels x 15 = 90 trials
trials_per_cell = 15

rows = []
for noise in sensory_uncertainty_levels:
    for intensity in stim_intensity_levels:
        for _ in range(trials_per_cell):
            rows.append({'sensory_uncertainty': noise, 'stim_intensity': intensity})

n_trials = len(rows)  # 90
assert n_trials == 90, f"Expected 90 trials, got {n_trials}"

df = pd.DataFrame(rows)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# --- Create independent check_up column (~1/6 ratio) ---
n_yes = 15
n_no = n_trials - n_yes  # 75

check_values = ['yes'] * n_yes + ['no'] * n_no
np.random.shuffle(check_values)

df['check_up'] = check_values

# Save file
filename = os.path.join(output_folder, "001_conditions_pilot.csv")
df.to_csv(filename, index=False)

print(f"Generated {filename}")
print(f"Total trials: {n_trials} (15 per cell, {len(sensory_uncertainty_levels)} noise x {len(stim_intensity_levels)} intensity levels)")
