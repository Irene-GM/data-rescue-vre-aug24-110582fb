from tools.helpers import get_filenames
from tools.image import read_image, save_image
from tools.calibrate import calibrate, get_calibration_settings, undistort_image
from tools.TableExtractor import TableExtractor
from tools.ManualSelect import ManualSelect
from tools.overlay import process_images
from tools.overlay_select_points import process_images_select_points

from pathlib import Path
import os
import logging


# Settings
CALIBRATE = False
CALIBRATION_DIR = "./calibration"
INPUT_DIR = "../data/st_eustatius"
CORRECTED_DIR = "../corrected"
OUTPUT_DIR = "../output/overlay"
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
    input_dir = Path.joinpath(Path(__file__).parent, INPUT_DIR).resolve().absolute()
    files = get_filenames(input_dir, extension="CR3")

    # Iterate over the file names
    for file in files:

        output_file = os.path.join(CORRECTED_DIR, os.path.relpath(os.path.splitext(file)[0] + ".png", input_dir))
        output_file_skip = os.path.splitext(output_file)[0] + ".skip"
        output_dir = os.path.join(CORRECTED_DIR, os.path.relpath(os.path.splitext(file)[0], input_dir))

        output_file = Path.joinpath(Path(__file__).parent, output_file).resolve().absolute()
        output_file_skip = Path.joinpath(Path(__file__).parent, output_file_skip).resolve().absolute()
        output_dir = Path.joinpath(Path(__file__).parent, output_dir).resolve().absolute()

        # Check whether image is already converted
        if (os.path.exists(output_file) or os.path.exists(output_file_skip)) and not OVERWRITE:
            continue

        logger.info(f"Converting image: '{file}'")

        # Load the image
        full_path = str(Path.joinpath(Path(__file__).parent, file).resolve())
        if not os.path.exists(full_path):
            raise OSError("Path does not exists")
        try:
            image = read_image(full_path)
        except Exception as e:
            logger.error(f"Reading image '{file}' failed: {e}")
            continue

        # Undistort image
        image = undistort_image(image, mtx, dist, newcameramtx, roi)

        # Manual detect image base
        selector = ManualSelect(image)
        try:
            image, corner_points = selector.detect()
        except Exception as e:
            logger.error(f"Selecting points for image: '{file}'")
            continue

        # Check whether manual selection was skipped
        if corner_points is None:
            if not os.path.exists(output_file_skip.parent):
                os.makedirs(output_file_skip.parent)
            with open(output_file_skip, "w") as f:
                f.write("Skip image\n")

            logger.warning("Skipped image corner point selection, image will not been shown next time")
            continue

        # Setup table extractor and detect the table base (largest contour)
        table_extractor = TableExtractor(image, output_dir, file)
        table_base = table_extractor.find_with_corners(corner_points)

        # Detect the table columns in the the table base
        #table_extractor = TableExtractor(table_base)
        #column_lines = table_extractor.find_with_lines(threshold_start=90)

        # Save image
        if table_base is not None:
            save_image(table_base, output_file)

    # Example usage for Overlay
    input_dir = CORRECTED_DIR
    output_dir = OUTPUT_DIR
    input_dir = Path.joinpath(Path(__file__).parent, input_dir).resolve().absolute()
    output_dir = Path.joinpath(Path(__file__).parent, output_dir).resolve().absolute()
    rows = 41
    columns = 1
    overlay_select_points = False

    # The user can select the start and the end of table. The code will draw lines based on the number of rows specified by user

    if overlay_select_points:
        num_lines = rows + 1
        process_images_select_points(input_dir, output_dir, num_lines, OVERWRITE)
    else:
        process_images(input_dir, output_dir, rows, columns, OVERWRITE)

if __name__ == "__main__":

    main()