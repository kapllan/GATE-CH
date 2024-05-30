import logging
import os
from datetime import datetime


def get_logger():
    current_path = os.path.abspath(os.getcwd())
    path_to_logs = list()

    # Get the path of the current script
    current_script_path = os.path.abspath(__file__)

    # Get the directory name of the current script
    current_script_directory = os.path.dirname(current_script_path)
    current_script_directory = current_script_directory.split('/')[-1]

    for sub_path in current_path.split('/'):
        if sub_path != current_script_directory:
            path_to_logs.append(sub_path)
        else:
            path_to_logs.append(current_script_directory)
            break
    path_to_logs.append('logs')
    path_to_logs = '/'.join(path_to_logs)
    if not os.path.exists(path_to_logs):
        os.makedirs(path_to_logs)

    # Configure logging
    logging.basicConfig(
        filename=f"{path_to_logs}/logs_{datetime.now().isoformat()}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(pathname)s - %(message)s'
    )

    return logging
