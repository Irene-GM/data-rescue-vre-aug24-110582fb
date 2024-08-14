
import os
import cv2
from fpdf import FPDF
import logging
from .helpers import get_filenames
from pathlib import Path



# Set up logging
import logging

logging.basicConfig()
logger = logging.getLogger("overlay_select_ponts")
logger.setLevel(logging.DEBUG)

# Mouse callback function to capture points
points = []

def select_points(event, x, y, flags, param):
    global points
    image_resized = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image_resized, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', image_resized)
        if len(points) == 2:
            cv2.destroyWindow('image')

def draw_lines_select_points(image_path, output_path, num_lines):
    global points
    points = []

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error: Unable to load image at {image_path}")
        return

    height, width = image.shape[:2]
    max_width = 1680
    max_height = 900

    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        image_resized = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
    else:
        image_resized = image.copy()

    cv2.imshow('image', image_resized)
    cv2.setMouseCallback('image', select_points, image_resized)
    cv2.waitKey(0)

    if len(points) == 2:
        y1 = int(points[0][1] * height / image_resized.shape[0])
        y2 = int(points[1][1] * height / image_resized.shape[0])
        spacing = abs(y2 - y1) // (num_lines - 1)

        for i in range(num_lines):
            y = min(y1, y2) + i * spacing
            cv2.line(image, (0, y), (width, y), (255, 0, 0), 1)

        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")
    else:
        logger.error("Error: Please select exactly two points.")

def process_images_select_points(input_dir, output_dir, num_lines, overwrite=False):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".png"):
                input_path = Path(root) / file
                relative_path = input_path.relative_to(input_dir)
                output_path = output_dir / relative_path

                if not output_path.parent.exists():
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                if not overwrite and output_path.exists():
                    logger.info(f"Skipping {output_path}, already exists.")
                    continue

                logger.info(f"Processing image: {input_path}")
                draw_lines_select_points(str(input_path), str(output_path), num_lines)

    # Get the overlayed image files
    files = get_filenames(output_dir, extension="png")
    sub_dirs = list(set([os.path.dirname(file) for file in files]))

    # Walk through the output directories
    for sub_dir in sub_dirs:

        # Get the overlayed image files in the parent directory
        files = get_filenames(sub_dir, extension="png")

        # Check whether there are files
        if len(files) < 1:
            continue
        
        # PDF path
        pdf_path = os.path.join(output_dir, os.path.basename(sub_dir), f"{os.path.basename(sub_dir)}.pdf")

        # Skip when PDF already exists
        if os.path.exists(pdf_path):
            continue

        # Initialize PDF file for subfolder
        pdf = FPDF()
        
        for filename in files:
            
            # Add the image to the PDF
            pdf.add_page()
            pdf.image(filename, x=0, y=0, w=210, h=297)  # A4 size in mm
        
        # Save the PDF file
        logger.info(f"Saving PDF: '{pdf_path}'")
        pdf.output(pdf_path)


