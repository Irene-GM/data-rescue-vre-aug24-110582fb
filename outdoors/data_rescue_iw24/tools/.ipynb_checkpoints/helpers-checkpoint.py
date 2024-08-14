from typing import List
import os

def get_filenames(path: str, extension: str = "") -> List[str]:
    """
    Collects filenames including the full path in the specified folder and returns a list of strings.

    :param path: Path to the folder containing the to be obtained files.
    :param extension: File extension for the files to find (e.g. '.json', '.csv', etc)
    :return: List of strings.
    """

    # Iterate over all files at the path
    filenames_filtered = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # The full file location
            full_path = os.path.join(dirpath, filename)

            # Check whether file is a 'extension' file
            if filename.endswith(extension):
                filenames_filtered.append(full_path)

    # Sort the filenames
    filenames_filtered.sort()

    return filenames_filtered