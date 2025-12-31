#!/usr/bin/env python
"""
Simple launcher script for the 2D visualization GUI.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visual_gui import main

if __name__ == "__main__":
    main()


