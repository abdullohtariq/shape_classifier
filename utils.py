import os

def create_directories(dir_list):
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)

# (optional) If you have any reusable functions, you can place them in utils.py.
