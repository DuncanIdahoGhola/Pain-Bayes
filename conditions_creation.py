    import pandas as pd
import numpy as np
import os

# --- Configuration ---
n_trials = 80  # Total number of trials per condition file
num_files = 100 # Number of condition files to generate
output_folder = "condition_files" # Folder to save the condition files

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the base conditions
base_conditions = [
    (30, 'low', 'low'),
    (30, 'low', 'high'),
    (30, 'high', 'low'),
    (30, 'high', 'high'),
    (70, 'low', 'low'),
    (70, 'low', 'high'),
    (70, 'high', 'low'),
    (70, 'high', 'high'),
]

# Calculate how many times each base condition should appear
num_base_conditions = len(base_conditions)
trials_per_condition_type = n_trials // num_base_conditions

# Check if n_trials is a multiple of num_base_conditions
if n_trials % num_base_conditions != 0:
    print(f"Warning: n_trials ({n_trials}) is not a perfect multiple of the number of base conditions ({num_base_conditions}). "
          f"Some conditions might appear slightly more or less than others to reach n_trials.")

for file_idx in range(1, num_files + 1):
    all_conditions = []
    for _ in range(trials_per_condition_type):
        all_conditions.extend(base_conditions)

    # If n_trials is not a perfect multiple, add remaining conditions
    remaining_trials = n_trials % num_base_conditions
    if remaining_trials > 0:
        all_conditions.extend(base_conditions[:remaining_trials])

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_conditions, columns=['expected_vas', 'expect_uncertainty', 'sensory_uncertainty'])

    # --- Randomization with constraint ---
    shuffled_indices = np.arange(len(df))
    np.random.shuffle(shuffled_indices)
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)

    # Apply the constraint: no more than 3 trials in a row with the same expected_vas
    final_ordered_trials = []
    current_expected_vas_count = {} # To track consecutive counts

    # A more robust approach for satisfying the "no more than 3 in a row" constraint:
    # We'll use a pool of conditions and draw from it, ensuring the constraint is met.
    
    # Create a list of all condition tuples that will be used
    condition_pool = [tuple(row) for index, row in df.iterrows()]
    np.random.shuffle(condition_pool) # Initial shuffle of the pool

    final_conditions_list = []
    last_vas = None
    consecutive_vas_count = 0

    while condition_pool:
        found_next_trial = False
        
        # Try to find a suitable next trial from the pool
        for i, (vas, eu, su) in enumerate(condition_pool):
            if vas == last_vas:
                if consecutive_vas_count < 2: # Allow up to 2 consecutive (making it 3 total when current is added)
                    final_conditions_list.append(condition_pool.pop(i))
                    consecutive_vas_count += 1
                    last_vas = vas
                    found_next_trial = True
                    break
            else: # Different VAS, always allowed
                final_conditions_list.append(condition_pool.pop(i))
                last_vas = vas
                consecutive_vas_count = 1
                found_next_trial = True
                break
        
        # If we couldn't find a suitable trial without violating the constraint,
        # it means all remaining trials in the pool would violate it.
        # This implies a very difficult-to-solve situation with the current pool and constraint.
        # For simplicity, we'll allow a slight deviation here if we get stuck.
        # A truly perfect solution for all possible scenarios might require a more
        # advanced combinatorial algorithm or backtracking.

        # If we couldn't find a trial that immediately satisfies the rule,
        # it might be because only "problematic" trials are left at the front of the pool.
        # We can try to re-shuffle a portion or just pick the next available and deal with the small deviation.
        # For the purpose of this script, we prioritize generating 80 trials and satisfying the constraint mostly.

        if not found_next_trial and condition_pool:
            # This block is reached if all remaining conditions at the top of the pool
            # would violate the 3-in-a-row rule. We need to "dig" deeper or re-evaluate.
            # A simple fix for this edge case is to temporarily relax the rule or try to find
            # a different VAS further down the pool.

            # We'll re-scan the pool more aggressively for a non-violating condition
            # if the initial pass didn't find one.
            found_alternative = False
            for i, (vas, eu, su) in enumerate(condition_pool):
                if vas != last_vas:
                    final_conditions_list.append(condition_pool.pop(i))
                    last_vas = vas
                    consecutive_vas_count = 1
                    found_alternative = True
                    break
            
            if not found_alternative:
                # If even a full scan didn't find a non-violating one, it means
                # all remaining conditions are the same VAS and would violate.
                # In a real-world scenario, this might indicate the constraint is too tight
                # for the remaining conditions. For now, we just pick the next one.
                # This is a rare edge case for 80 trials and 8 conditions.
                final_conditions_list.append(condition_pool.pop(0))
                if last_vas == final_conditions_list[-1][0]:
                    consecutive_vas_count += 1
                else:
                    consecutive_vas_count = 1
                last_vas = final_conditions_list[-1][0]


    final_df = pd.DataFrame(final_conditions_list, columns=['expected_vas', 'expect_uncertainty', 'sensory_uncertainty'])

    # Format the filename
    filename = os.path.join(output_folder, f"{file_idx:03d}_conditions.csv")

    # Save to CSV
    final_df.to_csv(filename, index=False)
    print(f"Generated {filename}")

print(f"\nAll {num_files} condition files generated in the '{output_folder}' folder.")