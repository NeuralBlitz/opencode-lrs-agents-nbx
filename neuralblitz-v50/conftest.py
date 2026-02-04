"""
Pytest configuration for NeuralBlitz V50.
Ensures proper import paths when running tests from any directory.
"""

import sys
import os

# Add the neuralblitz-v50 directory to the Python path
# This allows imports like 'from neuralblitz.minimal import ...' to work
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Verify the path is set up correctly
if project_root not in sys.path:
    sys.path.insert(0, project_root)
