#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PsychoPy calibration task using DS5 zero-uncertainty pulse trains.

Flow:
1. Hardware ready screen.
2. Three ascending ramps. Each ramp starts at START_MA and increases by
   INCREMENT_MA until the participant reports "no more".
3. Second calibration phase: selected fixed intensities are delivered and rated
   on a 0-100 pain scale.

Keys:
    p      experimenter confirms hardware ready
    space  deliver the currently displayed stimulation
    y      participant can continue / stimulation was tolerable
    n      no more / stop the current ramp
    left/right adjust rating marker
    return confirm rating
    escape quit safely
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from psychopy import core, data, event, gui, visual

from ds5_none_uncertainty import DS5PulseConfig, fire_none_uncertainty_pulse


expName = "PainCalibrationNoneUncertainty"
expInfo = {
    "participant": "sub-000",
    "start_ma": "1.0",
    "increment_ma": "1.0",
    "n_ramps": "3",
    "max_ma": "25.0",
    "v_per_ma": "0.4",
    "pulse_width_ms": "2",
    "pulse_frequency_hz": "100",
    "train_duration_ms": "500",
    "record_monitor": "True",
    "date": data.getDateStr(),
}


def get_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}


def check_escape():
    if "escape" in event.getKeys(["escape"]):
        core.quit()


def wait_for_key(valid_keys):
    while True:
        check_escape()
        keys = event.waitKeys(keyList=list(valid_keys) + ["escape"])
        if not keys:
            continue
        if keys[0] == "escape":
            core.quit()
        return keys[0]


def show_text(win, text, height=0.055):
    stim = visual.TextStim(
        win=win,
        text=text,
        color="white",
        height=height,
        wrapWidth=1.45,
        font="Open Sans",
    )
    stim.draw()
    win.flip()


def collect_rating(win, prompt, start_value=None):
    value = int(start_value if start_value is not None else np.random.randint(20, 81))
    value = max(0, min(100, value))

    prompt_stim = visual.TextStim(
        win=win,
        text=prompt,
        color="white",
        height=0.055,
        pos=(0, 0.28),
        wrapWidth=1.45,
    )
    label_left = visual.TextStim(
        win=win, text="Aucune\ndouleur", color="white", height=0.04, pos=(-0.55, -0.18)
    )
    label_right = visual.TextStim(
        win=win,
        text="Pire douleur\nimaginable",
        color="white",
        height=0.04,
        pos=(0.55, -0.18),
    )
    line = visual.Line(win=win, start=(-0.55, 0), end=(0.55, 0), lineColor="white")
    marker = visual.Rect(win=win, width=0.012, height=0.12, fillColor="white", lineColor="white")
    value_text = visual.TextStim(win=win, text="", color="white", height=0.05, pos=(0, -0.34))

    clock = core.Clock()
    event.clearEvents()
    while True:
        check_escape()
        for key in event.getKeys(["left", "right", "return", "num_enter", "escape"]):
            if key == "escape":
                core.quit()
            if key == "left":
                value = max(0, value - 1)
            elif key == "right":
                value = min(100, value + 1)
            elif key in {"return", "num_enter"}:
                return value, clock.getTime()

        marker.pos = (-0.55 + (value / 100.0) * 1.10, 0)
        value_text.text = str(value)
        prompt_stim.draw()
        line.draw()
        marker.draw()
        label_left.draw()
        label_right.draw()
        value_text.draw()
        win.flip()


def save_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    dlg = gui.DlgFromDict(expInfo, title=expName, sortKeys=False)
    if not dlg.OK:
        core.quit()

    start_ma = float(expInfo["start_ma"])
    increment_ma = float(expInfo["increment_ma"])
    n_ramps = int(expInfo["n_ramps"])
    max_ma = float(expInfo["max_ma"])
    output_dir = Path("data") / f"{expInfo['participant']}_{expInfo['date']}" / "calibration"

    config = DS5PulseConfig(
        v_per_ma=float(expInfo["v_per_ma"]),
        pulse_width_ms=float(expInfo["pulse_width_ms"]),
        pulse_frequency_hz=float(expInfo["pulse_frequency_hz"]),
        train_duration_ms=float(expInfo["train_duration_ms"]),
        record_monitor=get_bool(expInfo["record_monitor"]),
    )

    win = visual.Window(
        size=[1920, 1080],
        fullscr=True,
        color=[0, 0, 0],
        units="height",
        monitor="testMonitor",
    )
    win.mouseVisible = False

    ramp_rows = []
    rating_rows = []
    global_clock = core.Clock()

    try:
        show_text(
            win,
            "Bienvenue.\n\nAssurez-vous que le DS5 est pret, que les electrodes sont placees, "
            "et que l'intensite de depart est securitaire.\n\nExperimentatrice: appuyez sur P.",
        )
        wait_for_key(["p"])

        show_text(
            win,
            "Calibration, partie 1.\n\nChaque rampe recommence a l'intensite minimale. "
            "Appuyez sur ESPACE pour envoyer la stimulation affichee.\n\n"
            "Ensuite: Y = continuer, N = arreter cette rampe.",
        )
        wait_for_key(["space"])

        ramp_stop_values = []
        last_tolerated_values = []

        for ramp_index in range(1, n_ramps + 1):
            current_ma = start_ma
            last_tolerated = None
            show_text(
                win,
                f"Rampe {ramp_index} / {n_ramps}\n\n"
                "La prochaine stimulation recommence au plus bas niveau.\n\n"
                "Appuyez sur ESPACE pour commencer.",
            )
            wait_for_key(["space"])

            step_index = 1
            while current_ma <= max_ma:
                show_text(
                    win,
                    f"Rampe {ramp_index} / {n_ramps}\n\n"
                    f"Intensite: {current_ma:.2f} mA\n\n"
                    "ESPACE = envoyer",
                )
                wait_for_key(["space"])

                label = f"ramp{ramp_index:02d}_step{step_index:02d}_{current_ma:.2f}mA"
                metadata = fire_none_uncertainty_pulse(
                    current_ma,
                    config,
                    output_dir=output_dir / "waveforms",
                    trial_label=label,
                )

                show_text(
                    win,
                    "Est-ce que vous pouvez continuer?\n\nY = oui, continuer\nN = non, arreter cette rampe",
                )
                response = wait_for_key(["y", "n"])
                row = {
                    "participant": expInfo["participant"],
                    "time": global_clock.getTime(),
                    "phase": "ramp",
                    "ramp": ramp_index,
                    "step": step_index,
                    "intensity_ma": current_ma,
                    "response": response,
                    "noise_condition": "none",
                    **metadata,
                }
                ramp_rows.append(row)

                if response == "n":
                    ramp_stop_values.append(current_ma)
                    break

                last_tolerated = current_ma
                current_ma = round(current_ma + increment_ma, 6)
                step_index += 1
            else:
                ramp_stop_values.append(max_ma)

            if last_tolerated is not None:
                last_tolerated_values.append(last_tolerated)

        tolerance_ma = float(np.min(ramp_stop_values))
        if last_tolerated_values:
            high_ma = float(np.mean(last_tolerated_values))
        else:
            high_ma = max(start_ma, tolerance_ma - increment_ma)
        low_ma = start_ma + 0.50 * max(0.0, high_ma - start_ma)

        show_text(
            win,
            "Calibration, partie 2.\n\n"
            "Vous allez maintenant evaluer quelques stimulations fixes sur une echelle de douleur.\n\n"
            "Appuyez sur ESPACE pour commencer.",
        )
        wait_for_key(["space"])

        candidate_intensities = [start_ma, low_ma, high_ma]
        candidate_intensities = sorted({round(x, 3) for x in candidate_intensities if x <= max_ma})

        trial_number = 1
        for repeat_index in range(1, 3):
            for intensity_ma in candidate_intensities:
                show_text(
                    win,
                    f"Evaluation {trial_number}\n\nIntensite: {intensity_ma:.2f} mA\n\nESPACE = envoyer",
                )
                wait_for_key(["space"])
                label = f"rating{trial_number:02d}_{intensity_ma:.2f}mA"
                metadata = fire_none_uncertainty_pulse(
                    intensity_ma,
                    config,
                    output_dir=output_dir / "waveforms",
                    trial_label=label,
                )
                rating, rt = collect_rating(
                    win,
                    "Veuillez evaluer l'intensite de la stimulation que vous venez de recevoir.",
                )
                rating_rows.append(
                    {
                        "participant": expInfo["participant"],
                        "time": global_clock.getTime(),
                        "phase": "rating",
                        "repeat": repeat_index,
                        "trial": trial_number,
                        "intensity_ma": intensity_ma,
                        "rating": rating,
                        "rating_rt": rt,
                        "noise_condition": "none",
                        **metadata,
                    }
                )
                trial_number += 1

        summary_rows = [
            {
                "participant": expInfo["participant"],
                "start_ma": start_ma,
                "increment_ma": increment_ma,
                "n_ramps": n_ramps,
                "max_ma": max_ma,
                "ramp_stop_values": ";".join(f"{x:.3f}" for x in ramp_stop_values),
                "last_tolerated_values": ";".join(f"{x:.3f}" for x in last_tolerated_values),
                "recommended_low_ma": round(low_ma, 3),
                "recommended_high_ma": round(high_ma, 3),
                "recommended_failsafe_ma": round(tolerance_ma, 3),
                "noise_condition": "none",
            }
        ]

        common_fields = [
            "participant",
            "time",
            "phase",
            "ramp",
            "step",
            "intensity_ma",
            "response",
            "noise_condition",
            "n_pulses",
            "samples_per_pulse",
            "samples_per_period",
            "samples_total",
            "max_requested_v",
        ]
        save_rows(output_dir / "ramp_data.csv", ramp_rows, common_fields)
        save_rows(
            output_dir / "rating_data.csv",
            rating_rows,
            [
                "participant",
                "time",
                "phase",
                "repeat",
                "trial",
                "intensity_ma",
                "rating",
                "rating_rt",
                "noise_condition",
                "n_pulses",
                "samples_per_pulse",
                "samples_per_period",
                "samples_total",
                "max_requested_v",
            ],
        )
        save_rows(
            output_dir / "summary.csv",
            summary_rows,
            [
                "participant",
                "start_ma",
                "increment_ma",
                "n_ramps",
                "max_ma",
                "ramp_stop_values",
                "last_tolerated_values",
                "recommended_low_ma",
                "recommended_high_ma",
                "recommended_failsafe_ma",
                "noise_condition",
            ],
        )

        show_text(
            win,
            f"Calibration terminee.\n\n"
            f"Low recommande: {low_ma:.2f} mA\n"
            f"High recommande: {high_ma:.2f} mA\n"
            f"Failsafe recommande: {tolerance_ma:.2f} mA\n\n"
            "Appuyez sur ESPACE pour quitter.",
        )
        wait_for_key(["space"])
    finally:
        win.close()
        os.chdir(Path(__file__).parent)


if __name__ == "__main__":
    main()
