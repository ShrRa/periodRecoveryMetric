import os
import shutil
def make_file(loc):
    try:
        os.mkdir(loc)
    except FileExistsError:
        shutil.rmtree(loc)
        os.mkdir(loc)
    return loc+'/'