import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, sin, cos, exp
import obspy
import importlib
import cleanbf
import subprocess

#%% check that the demos run without error
def test_demo_real():
    #import demo_real_data
    result = subprocess.run(["python", "demos/demo_real_data.py"], capture_output=True, text=True)
    assert result.returncode == 0

    
def test_demo_synth():
    result = subprocess.run(["python", "demos/demo_synthetic.py"], capture_output=True, text=True)
    assert result.returncode == 0
    #import demo_synthetic

