import os
import time
import sys
from contextlib import contextmanager


@contextmanager
def silence():
    old_target_out = sys.stdout
    old_target_err = sys.stderr
    
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            sys.stderr = new_target
            yield new_target
    finally:
        sys.stdout = old_target_out
        sys.stderr = old_target_err
        

def create_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

        
def fit_and_time(fitclass):
    def timed(*args, **kw):
        ts = time.time()
        result = fitclass.fit(*args, **kw)
        te = time.time()
        fitclass.time_ = te - ts
        return result
    return timed
