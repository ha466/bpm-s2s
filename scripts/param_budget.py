"""Print exact parameter counts + estimated memory/FLOPs for a config. STATUS: working code.
    python scripts/param_budget.py configs/ultra_small.yaml
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bpm.config import load_cfg
from bpm.budget import budget
m, _ = load_cfg(sys.argv[1])
print(json.dumps(budget(m), indent=2))
