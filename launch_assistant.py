import subprocess
import os
import sys

script_path = os.path.abspath("simple_hybrid_cli.py")  # full path to your script

subprocess.Popen(
    ["powershell.exe", "-NoExit", "-Command", f"python '{script_path}'"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
