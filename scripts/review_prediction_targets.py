"""Common dev evaluation and paired generation for the two completed velocity targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_continuation import main


if __name__=="__main__":
    main(arms=("v","v_x0loss"),default_folder="runs/prediction_target_review_20261101",
         evidence_root="docs/experiments/prediction-target")
