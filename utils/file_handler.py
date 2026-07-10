import os
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
