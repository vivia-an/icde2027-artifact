"""Starts the Megatron GPU job. The 36-cell run lives in launch_megatron.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(HERE, "launch_megatron.py")
os.execv(sys.executable, [sys.executable, script, *sys.argv[1:]])
