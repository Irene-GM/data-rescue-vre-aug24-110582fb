from tools.helpers import get_filenames
from tools.image import read_image, save_image
from tools.calibrate import calibrate, get_calibration_settings, undistort_image
from tools.TableExtractor import TableExtractor

from pathlib import Path
import os
import logging


# Settings
CALIBRATE = False
CALIBRATION_DIR = "./calibration"
INPUT_DIR = "./data/st_eustatius"
OUTPUT_DIR = "./corrected/st_eustatius"
OVERWRITE = False

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():

    # Perform camera calibration
    calibration_dir = Path.joinpath(Path(__file__).parent, CALIBRATION_DIR).resolve()
    if CALIBRATE:
        mtx, dist, newcameramtx, roi = calibrate(calibration_dir)
        logger.info("Calibration done, make sure to disable CALIBRATE")
     
    mtx, dist, newcameramtx, roi = get_calibration_settings(calibration_dir)

    # Get the image files
    #files = get_filenames(INPUT_DIR, extension="CR3")

    # Test for only one file
    files = ["./data/st_eustatius/1910/2024-06-04_12-32-41_0001.CR3"]
    #files = ["./data/st_eustatius/1911/IMG_0052.CR3"]

    # Iterate over the file names
    for file in files:

        output_file = os.path.join(OUTPUT_DIR, os.path.relpath(os.path.splitext(file)[0] + ".png", INPUT_DIR))
        output_dir = os.path.join(OUTPUT_DIR, os.path.relpath(os.path.splitext(file)[0], INPUT_DIR))

        output_file = Path.joinpath(Path(__file__).parent, output_file).resolve()
        output_dir = Path.joinpath(Path(__file__).parent, output_dir).resolve()

        # Check whether image is already converted
        if os.path.exists(output_file) and not OVERWRITE:
            continue

        logger.info(f"Converting image: '{file}'")

        # Load the image
        try:
            image = read_image(file)
        except Exception as e:
            logger.error(f"Reading image '{file}' failed: {e}")
            continue

        # Undistort image
        image = undistort_image(image, mtx, dist, newcameramtx, roi)

        # Setup table extractor and detect the table base (largest contour)
        table_extractor = TableExtractor(image, output_dir, file)
        table_base = table_extractor.find_with_min_area(threshold_start=40, threshold_end=120, area_percentage_min=0.45, area_percentage_max=0.85)

        # Detect the table columns in the the table base
        #table_extractor = TableExtractor(table_base)
        #column_lines = table_extractor.find_with_lines(threshold_start=90)

        # Save image
        if table_base is not None:
            save_image(table_base, output_file)

if __name__ == "__main__":

    main()