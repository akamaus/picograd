import platform
import os

sys = platform.system()

def link(origin, link_path):
    if sys == "Windows":
        import win32file
        win32file.CreateSymbolicLink(link_path, origin)
    else:
        os.link(origin, link_path)
