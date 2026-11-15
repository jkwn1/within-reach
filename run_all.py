"""Run the whole pipeline, raw data -> every number and figure in the slides.

Usage:  python run_all.py            (all steps)
        python run_all.py 3 4 5 5b 6 8 9   (only some steps, e.g. after editing the model)

Expects the raw files listed in data/raw/README.md. Total runtime is about 30 minutes on a 2-core machine;
most of it is the hyperparameter search in step 3 (about 25 minutes).
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = {
    "0": "00_ingest.py",            # raw CSV/Excel/Gazetteer -> data/interim/*.parquet
    "1": "01_job_access.py",        # Act 1: entry-level pay within reach of each Bay Area tract vs. mobility
    "2": "02_college_prepare.py",   # one row per college, target + inputs + affordability join
    "3": "03_college_model.py",     # baselines, tuned model, held-out test, cross-fitted value-added
    "4": "04_college_fairness.py",  # errors by group, bottom-20% comparison, race-input sensitivity
    "5": "05_opportunity.py",       # opportunity colleges, hidden gems, student-parent gap
    "5b": "05b_choice.py",          # similar-students spread, "also consider" options, independent-student test
    "6": "06_figures.py",           # every chart in the slides
    "7": "07_handcheck_sheets.py",  # random samples for the manual checks
    "8": "08_key_numbers.py",       # results/key_numbers.md: every number quoted in the slides
    "9": "09_finder.py",            # docs/index.html: the Within Reach College Finder web page
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(STEPS)
    for step in chosen:
        t0 = time.time()
        print(f"\n=== step {step}: {STEPS[step]}", flush=True)
        subprocess.run([sys.executable, str(SRC / STEPS[step])], check=True)
        print(f"=== step {step} done in {time.time() - t0:.0f}s", flush=True)
