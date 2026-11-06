"""Train a randomly initialized local byte AE on UTF-8 windows."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.ae_training import train
from char_diff.utils import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("config", nargs="?", default="configs/tinystories_multilingual_ae48.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--resume", help="Complete AE checkpoint, including optimizer and sampler")
    p.add_argument("--cooldown-from", help="Branch from a stable full checkpoint into a new run directory")
    p.add_argument("--device")
    p.add_argument("--stop-after", type=int, default=0, help="Successful updates in this invocation; 0 = run_bytes limit")
    p.add_argument("--stop-at-bytes", type=int, default=0, help="Pause at a successful-byte boundary without changing the schedule")
    args = p.parse_args()
    train(load_config(args.config, args.set), resume=args.resume, device=args.device,
          stop_after=args.stop_after, stop_at_bytes=args.stop_at_bytes, cooldown_from=args.cooldown_from)
