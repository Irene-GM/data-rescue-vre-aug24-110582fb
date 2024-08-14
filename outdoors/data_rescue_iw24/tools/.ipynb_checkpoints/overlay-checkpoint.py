import os
import cv2
from fpdf import FPDF
import logging
from .helpers import get_filenames
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger("overlay")
logger.setLevel(logging.DEBUG)

def draw_table(image_path, rows, columns, output_path):

    # Load the image using OpenCV
    img = cv2.imread(image_path)
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate cell dimensions
    cell_width = width // columns
    cell_height = height // rows
    
    # Draw horizontal lines
    for i in range(1, rows):
        cv2.line(img, (0, i * cell_height), (width, i * cell_height), (255, 255, 255), 2)
    
    # Draw vertical lines
    for i in range(1, columns):
        cv2.line(img, (i * cell_width, 0), (i * cell_width, height), (0, 0, 0), 1)
    
    # Save the modified image
    cv2.imwrite(output_path, img)

def process_images(input_dir, output_dir, rows, columns, overwrite=False):
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the image files
    files = get_filenames(input_dir, extension="png")
    
    # Walk through the input directory
    for file in files:
    
        output_file = os.path.join(output_dir, os.path.relpath(os.path.splitext(file)[0] + ".png", input_dir))
        output_file = Path.joinpath(Path(__file__).parent, output_file).resolve().absolute()

        # Check whether image is already converted
        if os.path.exists(output_file) and not overwrite:
            continue

        logger.info(f"Overlaying image: '{file}'")

        # Make sure output directory exists
        if not os.path.exists(output_file.parent):
            os.makedirs(output_file.parent)

        # Draw table on the image and save it
        draw_table(file, rows, columns, output_file)

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
        pdf_path = os.path.join(output_dir, f"{os.path.basename(sub_dir)}.pdf")

        # Skip when PDF already exists
        if os.path.exists(pdf_path):
            continue

        # Initialize PDF file for subfolder
        pdf = FPDF()
        
        for filename in sorted(files):
            
            # Add the image to the PDF
            pdf.add_page()
            pdf.image(filename, x=0, y=0, w=210, h=297)  # A4 size in mm
            
        # Save the PDF file
        logger.info(f"Saving PDF: '{pdf_path}'")
        pdf.output(pdf_path)

