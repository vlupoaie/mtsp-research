import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

INPUTS_DIR = os.path.join(ROOT_DIR, 'inputs')

OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
if not os.path.isdir(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)
