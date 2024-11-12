## Data Rescue of Dutch historic weather journals: sharing the basics

### Context

Every year the department of Observations & Data Technology (RDWD) organizes the Innovation Weeks (IW), which is a 2-week sprint (hackathon) bringing researchers and developers together to work full-time on innovative ideas. In the IW24 we formed a teamt to work in the digitization of Dutch historic weather journals for the Caribbean Islands (i.e. Bonaire, Curacao, St. Eustatius, St. Maarten). Part of the team spent their time at the KNMI's archive taking photos of the original documents, while another part of the team was busy trying different digitization approaches enabling the automatic extraction of the measurements. These coding activities were carried out in a virtual research environment (VRE) platform equipped with an additional data provenance engine (SWIRRL; [https://gitlab.com/KNMI-OSS/swirrl/swirrl-api](https://gitlab.com/KNMI-OSS/swirrl/swirrl-api)), where users could share data and code. Thus, this public Github repository is a snapshot version of the VRE used during the IW24, but simplified to ease the comprehension of its contents. 

### Overview on the analytical work done

In a nutshell, during the IW24 we explored the following aspects of data rescue:

1. Digitization of handwritten weather journals with [SMHI's Dawsonia](https://git.smhi.se/ai-for-obs/dawsonia)
2. Digitization of typewrited weather journals with optical character recognition (OCR)
3. Manual labeling and object recognition using deep learning
4. Comparison of historic records with reanalysis data for a severe storm in 1924

The format of the historic handwritten Dutch weather journals can be summarized as *"compact and dense horizontal tables (wide format) with faint gridlines"*. This posed challenges for `dawsonia`'s table extraction software as it is difficult to identify where the tables (and cells) are located. This prompted development of additional image pre-processing steps, and a customized method to detect the structure of the table (e.g. pre-set rows and columns). By doing this process ourselves, we managed to get `dawsonia` to identify some of the digits in the documents for a few months in St. Eustatius and St. Maarten. The weather journals for Bonaire are typewrited (at least a fraction of them), so we also tried an OCR approach with Python's `easyocr` library. This approach seems to identify digits for our example in Bonaire in April 1927. 

A different research line that we tried opted by manually labelling some documents with VIA ([VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)) and then using a pre-trained deep learning model for object detection and image segmentation ([Ultralytics' YOLO v8](https://docs.ultralytics.com/)). This required quite some effort at manually labelling the digits for a few journal pages, so that they can be later identified with YOLO. This showed some skill at identifying digits, but we ran out of time for a deeper testing. 

Finally, we manually digitized a journal page for St. Eustatius and St. Maarten for August 1924. In this month, a severe storm crossed the region causing a lot of damage in the region. We compared the daily measurements against reanalysis data from ERA5, to get a picture on the potential added value of these historical measurments. 


### What can I find in this repo?

The `outdoors` folder contains the necessary code and data to reproduce the results obtained during the IW24. It is organized in 3 subfolders:

- `data_rescue_iw24`: Contains Python scripts developed to do the table detection customized to the Dutch weather journals. The file `TableExtractor.py` contains most of the work, since it will extract the table with the historic measurements and apply a set of image corrections intended to ease the detection of digits with `dawsonia` down the line. Note that we also developed a manual method in which a user is prompted with a journal image to manually identify the corners of the table. This extension to the original effort can be found in the file `ManualSelect.py`. The process of cropping and correcting images is better illustrated in `notebooks/image-correction.ipynb`.
- `notebooks`: Contains the Jupyter notebooks developed during the IW24. The notebooks `automatic-overlay.ipynb` and `image-correction.ipynb` illustrate the table detection process for handwritten journals (item #1). Then `easy-ocr-test.ipynb` shows an example with typewrited text (item #2). The file `yolov8-testing.ipynb` shows how to do object detection after doing the manual digitization with VIA. Then, the file `reanalysis_data.ipynb` visualizes the historic observations against reanalysis data from ERA5. The two subfolders in `notebooks/` provide a space for the scripts to write some outputs. 
- `sample_data`: As the name indicates, this folder contains the necessary datasets to run the notebooks. The subfolder `logbooks` contains a few samples of the weather journals and the image processing. Then `reanalysis_example` contains the manually digitized measurements, originally as Excel spreadsheets and then transformed into NetCDF files for the data visualization part. 

