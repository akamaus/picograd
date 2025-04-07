import platform
import os
import shutil

sys = platform.system()

def link(origin, link_path):
    if sys == "Windows":
        shutil.copy(origin, link_path)
    else:
        os.link(origin, link_path)
