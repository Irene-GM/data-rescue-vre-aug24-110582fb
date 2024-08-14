import cv2
import numpy as np
import logging
import math
import os
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger("TableExtractor")
logger.setLevel(logging.DEBUG)

class TableExtractor:

    def __init__(self, image, output_dir, file):
        self.image = image
        self.output_dir = output_dir
        self.file = file
        self.filename = os.path.splitext(os.path.basename(file))[0] + ".jpg"
        self.threshold = 0

    def find_with_corners(self, corner_points):

        self.mean_width = 100

        self.store_process_image(f"0_original_{self.filename}", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image(f"1_grayscaled_{self.filename}", self.grayscale_image)
        self.convolve_image(corner_points=corner_points)
        self.store_process_image(f"2_convolved_{self.filename}", self.convolved_image)
        self.order_points_in_the_contour_with_max_area(corner_points)
        self.store_process_image(f"3_with_4_corner_points_plotted_{self.filename}", self.image_with_points_plotted)
        self.calculate_new_width_and_height_of_image()
        self.apply_perspective_transform()
        self.store_process_image(f"4_perspective_corrected_{self.filename}", self.perspective_corrected_image)
        
        return self.perspective_corrected_image
        

    def find_with_min_area(self, threshold_start = 30, threshold_end=150, area_percentage_min=0.5, area_percentage_max=0.85):

        self.threshold = threshold_start
        self.threshold_end = threshold_end
        self.area_largest_contour = 0
        self.area_percentage = 0
        self.area_percentage_min = area_percentage_min
        self.area_percentage_max = area_percentage_max
        self.mean_width = 100

        self.store_process_image(f"0_original_{self.filename}", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image(f"1_grayscaled_{self.filename}", self.grayscale_image)
        self.convolve_image()
        self.store_process_image(f"2_convolved_{self.filename}", self.convolved_image)
        self.find_borders()
        self.store_process_image(f"3_bordered_{self.filename}", self.bordered_image)

        while self.area_percentage < self.area_percentage_min:

            self.threshold_image()
            self.store_process_image(f"4_thresholded_{self.filename}", self.thresholded_image)
            self.invert_image()
            self.store_process_image(f"5_inverteded_{self.filename}", self.inverted_image)
            self.dilate_image()
            self.store_process_image(f"6_dialateded_{self.filename}", self.dilated_image)
            self.find_contours()
            self.store_process_image(f"7_all_contours_{self.filename}", self.image_with_all_contours)
            self.filter_contours_and_leave_only_rectangles()
            self.store_process_image(f"8_only_rectangular_contours_{self.filename}", self.image_with_only_rectangular_contours)
            self.find_largest_contour_by_area()
            self.store_process_image(f"9_contour_with_max_area_{self.filename}", self.image_with_contour_with_max_area)

            self.area_percentage = self.area_largest_contour / self.grayscale_image.size

            logger.debug(f"Finding threshold ({self.threshold}) --> area percentage: {self.area_percentage * 100.0}%")

            if self.threshold+1 >= self.threshold_end:
                logger.error(f"Image '{self.file}' could not be corrected (maximum threshold achieved)")
                return None
            
            if self.area_percentage > self.area_percentage_max:
                logger.error(f"Image '{self.file}' could not be corrected (maximum area percentage achieved)")
                return None
            
            self.threshold += 1
            
        self.order_points_in_the_contour_with_max_area()
        self.store_process_image(f"10_with_4_corner_points_plotted_{self.filename}", self.image_with_points_plotted)
        self.calculate_new_width_and_height_of_image()
        self.apply_perspective_transform()
        self.store_process_image(f"11_perspective_corrected_{self.filename}", self.perspective_corrected_image)
        
        return self.perspective_corrected_image
    
    def find_with_lines(self, threshold_start = 50, line_percentage_min=0.8):

        self.threshold = threshold_start
        self.column_lines = []

        self.store_process_image("0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image("1_grayscaled.jpg", self.grayscale_image)

        min_line_length = int(self.grayscale_image.shape[0] * line_percentage_min)
        max_line_gap = int((self.grayscale_image.shape[0] * line_percentage_min) / 50)

        while self.threshold < 100:

            self.threshold_image()
            self.store_process_image("3_thresholded.jpg", self.thresholded_image)
            self.invert_image()
            self.store_process_image("4_inverteded.jpg", self.inverted_image)
            self.dilate_image()
            self.store_process_image("5_dialateded.jpg", self.dilated_image)
            self.find_vertical_lines(min_line_length=min_line_length, max_line_gap=max_line_gap)
            self.store_process_image("6_all_lines.jpg", self.image_with_all_lines)

            logger.debug(f"Finding threshold ({self.threshold}) --> number of lines: {len(self.column_lines)}")

            self.threshold += 2
        
        return self.column_lines
    
    def find_with_points(self, threshold_start = 75, x_point=0, y_point=0, area_percentage_max=0.5):

        self.threshold = threshold_start
        self.x_point = -1
        self.y_point = -1
        self.area_percentage = 1.0
        self.area_percentage_max = area_percentage_max

        self.store_process_image("0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image("1_grayscaled.jpg", self.grayscale_image)

        self.area_largest_contour = self.grayscale_image.size

        while (self.x_point != x_point and self.y_point != y_point) or self.area_percentage > self.area_percentage_max:

            self.threshold_image()
            self.store_process_image("3_thresholded.jpg", self.thresholded_image)
            self.invert_image()
            self.store_process_image("4_inverteded.jpg", self.inverted_image)
            self.dilate_image()
            self.store_process_image("5_dialateded.jpg", self.dilated_image)
            self.find_contours()
            self.store_process_image("6_all_contours.jpg", self.image_with_all_contours)
            self.filter_contours_and_leave_only_rectangles()
            self.store_process_image("7_only_rectangular_contours.jpg", self.image_with_only_rectangular_contours)
            self.find_contour_by_points(x_point, y_point)

            self.area_percentage = self.area_largest_contour / self.grayscale_image.size

            logger.debug(f"Finding threshold ({self.threshold}) --> points: ({self.x_point}, {self.y_point}), area percentage: {self.area_percentage * 100.0}%")

            self.threshold += 2
            
        self.store_process_image("8_contour_with_x_y.jpg", self.image_with_contour_with_max_area)
        self.order_points_in_the_contour_with_max_area()
        self.store_process_image("9_with_4_corner_points_plotted.jpg", self.image_with_points_plotted)
        self.calculate_new_width_and_height_of_image()
        self.apply_perspective_transform()
        self.store_process_image("10_perspective_corrected.jpg", self.perspective_corrected_image)
        
        return self.perspective_corrected_image, self.contour_with_max_area_ordered

    def convert_image_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def convolve_image(self, corner_points=[]):

        # Determine max mean_width using corner_points
        if len(corner_points) > 0:

            tl = corner_points[0]
            if tl[0] < self.mean_width*2:
                self.mean_width = int((tl[0] / 2))
            if tl[1] < self.mean_width*2:
                self.mean_width = int((tl[1] / 2))

            tr = corner_points[1]
            if self.grayscale_image.shape[1] - tr[0] < self.mean_width*2:
                self.mean_width = int(((self.grayscale_image.shape[1] - tr[0]) / 2))
            if tr[1] < self.mean_width*2:
                self.mean_width = int((tr[1] / 2))

            br = corner_points[2]
            if self.grayscale_image.shape[1] - br[0] < self.mean_width*2:
                self.mean_width = int(((self.grayscale_image.shape[1] - br[0]) / 2))
            if self.grayscale_image.shape[0] - br[1] < self.mean_width*2:
                self.mean_width = int(((self.grayscale_image.shape[0] - br[1]) / 2))

            bl = corner_points[3]
            if bl[0] < self.mean_width*2:
                self.mean_width = int((bl[0] / 2))
            if self.grayscale_image.shape[0] - bl[1] < self.mean_width*2:
                self.mean_width = int(((self.grayscale_image.shape[0] - bl[1]) / 2))

        # Determine the horizontal mean
        image_horizontal_mean = np.mean(self.grayscale_image, axis=0)
        image_horizontal_convolve = np.convolve(image_horizontal_mean, np.ones(self.mean_width)/self.mean_width, mode='same')
        image_horizontal_corrected = self.grayscale_image[self.mean_width:self.grayscale_image.shape[0]-self.mean_width] / image_horizontal_convolve
        image_horizontal_corrected = 255 * (image_horizontal_corrected - image_horizontal_corrected.min()) / (image_horizontal_corrected.max() - image_horizontal_corrected.min())

        image_vertical_mean = np.mean(image_horizontal_corrected, axis=1)
        image_vertical_convolve = np.convolve(image_vertical_mean, np.ones(self.mean_width)/self.mean_width, mode='same')
        cropped_image = image_horizontal_corrected[:, self.mean_width:image_horizontal_corrected.shape[1]-self.mean_width]
        image_vertical_corrected = cropped_image / np.expand_dims(image_vertical_convolve, axis=1)
        image_vertical_corrected = 255 * (image_vertical_corrected - image_vertical_corrected.min()) / (image_vertical_corrected.max() - image_vertical_corrected.min())

        self.convolved_image = image_vertical_corrected.astype(np.uint8)
        self.convolved_image = self.convolved_image[self.mean_width:self.convolved_image.shape[0]-self.mean_width, self.mean_width:self.convolved_image.shape[1]-self.mean_width]

        self.image = self.image[self.mean_width*2:self.image.shape[0]-self.mean_width*2, self.mean_width*2:self.image.shape[1]-self.mean_width*2, :]

    def find_borders(self):

        image_horizontal_mean = np.mean(self.convolved_image, axis=0)
        image_mean = np.mean(image_horizontal_mean)
        image_std = 2*np.std(image_horizontal_mean)

        upper_limit = image_mean+image_std
        lower_limit = image_mean-image_std

        left_peak = 0
        for i, value in enumerate(image_horizontal_mean):
            if value > upper_limit or value < lower_limit:
                left_peak = i
                break

        right_peak = self.convolved_image.shape[1]
        for i, value in enumerate(np.flip(image_horizontal_mean)):
            if value > upper_limit or value < lower_limit:
                right_peak = self.convolved_image.shape[1] - i 
                break

        #plt.plot(image_horizontal_mean)
        #plt.plot([upper_limit]*len(image_horizontal_mean))
        #plt.plot([lower_limit]*len(image_horizontal_mean))
        #plt.show()

        image_vertical_mean = np.mean(self.convolved_image, axis=1)
        image_mean = np.mean(image_vertical_mean)
        image_std = 2*np.std(image_vertical_mean)

        upper_limit = image_mean+image_std
        lower_limit = image_mean-image_std

        top_peak = 0
        for i, value in enumerate(image_vertical_mean):
            if value > upper_limit or value < lower_limit:
                top_peak = i
                break

        bottom_peak = self.convolved_image.shape[0]
        for i, value in enumerate(np.flip(image_vertical_mean)):
            if value > upper_limit or value < lower_limit:
                bottom_peak = self.convolved_image.shape[0] - i 
                break

        #plt.plot(image_vertical_mean)
        #plt.plot([upper_limit]*len(image_vertical_mean))
        #plt.plot([lower_limit]*len(image_vertical_mean))
        #plt.show()

        self.image = self.image[top_peak:bottom_peak, left_peak:right_peak, :]
        self.bordered_image = self.convolved_image[top_peak:bottom_peak, left_peak:right_peak]
        

    def blur_image(self):
        self.blurred_image = cv2.blur(self.grayscale_image, (5, 5))

    def threshold_image(self):

        #self.thresholded_image = cv2.threshold(self.grayscale_image,self.threshold,255,cv2.THRESH_BINARY)[1]

        blur = cv2.GaussianBlur(self.bordered_image,(5,5),0)
        self.thresholded_image = cv2.threshold(blur,self.threshold,255,cv2.THRESH_BINARY)[1]

        #self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def dilate_image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

    def find_contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.image.copy()
        cv2.drawContours(self.image_with_all_contours, self.contours, -1, (0, 255, 0), 3)

    def find_vertical_lines(self, min_line_length, max_line_gap, max_horizontal_diff = 50):
        self.lines = cv2.HoughLines(self.dilated_image, 1, np.pi / 180, self.threshold, None, 0, 0)
        self.image_with_all_lines = self.image.copy()
        if self.lines is None:
            return
        
        self.column_lines = []
        for line in self.lines:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            horizontal_diff = abs(x2 - x1)
            line_length = abs(y2 - y1)

            if horizontal_diff < max_horizontal_diff:

                self.column_lines.append(line)            
                cv2.line(self.image_with_all_lines, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

        

    def filter_contours_and_leave_only_rectangles(self, epsilon_angle=3):
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:

                ordered_points = self.order_points(approx)

                tl = ordered_points[0]
                tr = ordered_points[1]
                br = ordered_points[2]
                bl = ordered_points[3]

                ba = bl - tl
                bc = tr - tl
                tl_denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
                if tl_denom == 0:
                    continue
                cosine_angle_tl = np.dot(ba, bc) / tl_denom
                angle_tl = np.degrees(np.arccos(cosine_angle_tl))

                ba = tl - tr
                bc = br - tr
                tr_denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
                if tr_denom == 0:
                    continue
                cosine_angle_tr = np.dot(ba, bc) / tr_denom
                angle_tr = np.degrees(np.arccos(cosine_angle_tr))

                ba = tr - br
                bc = bl - br
                br_denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
                if br_denom == 0:
                    continue
                cosine_angle_br = np.dot(ba, bc) / br_denom
                angle_br = np.degrees(np.arccos(cosine_angle_br))

                ba = br - bl
                bc = tl - bl
                bl_denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
                if bl_denom == 0:
                    continue
                cosine_angle_bl = np.dot(ba, bc) / bl_denom
                angle_bl = np.degrees(np.arccos(cosine_angle_bl))

                if angle_tl > (90 - epsilon_angle) and angle_tl < (90 + epsilon_angle) and \
                    angle_tr > (90 - epsilon_angle) and angle_tr < (90 + epsilon_angle) and \
                    angle_br > (90 - epsilon_angle) and angle_br < (90 + epsilon_angle) and \
                    angle_bl > (90 - epsilon_angle) and angle_bl < (90 + epsilon_angle):

                        self.rectangular_contours.append(approx)
                    
        self.image_with_only_rectangular_contours = self.image.copy()
        cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

    def find_largest_contour_by_area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour

        self.area_largest_contour = max_area
        self.image_with_contour_with_max_area = self.image.copy()
        cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)

    def find_contour_by_points(self, x, y, epsilon=50):
        self.contour_with_max_area = None
        min_area = self.image.size
        for contour in self.rectangular_contours:

            area = cv2.contourArea(contour)
            contour_with_max_area_ordered = self.order_points(contour)

            for point in contour_with_max_area_ordered:
                
                if (point[0] > x - epsilon) and (point[0] < x + epsilon) and (point[1] > y - epsilon) and (point[1] < y + epsilon):
                    if area < min_area:

                        self.x_point = point[0]
                        self.y_point = point[1]
                        self.contour_with_max_area = contour
                        min_area = area
        
        if self.contour_with_max_area is not None:
            self.area_largest_contour = min_area
            self.image_with_contour_with_max_area = self.image.copy()
            cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)


    def order_points_in_the_contour_with_max_area(self, corner_points = []):
        if len(corner_points) < 1:
            self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)
        else:
            self.contour_with_max_area_ordered = self.order_points(corner_points)
            self.contour_with_max_area_ordered[0][0] = self.contour_with_max_area_ordered[0][0] - self.mean_width*2
            self.contour_with_max_area_ordered[0][1] = self.contour_with_max_area_ordered[0][1] - self.mean_width*2
            self.contour_with_max_area_ordered[1][0] = self.contour_with_max_area_ordered[1][0] - self.mean_width*2
            self.contour_with_max_area_ordered[1][1] = self.contour_with_max_area_ordered[1][1] - self.mean_width*2
            self.contour_with_max_area_ordered[2][0] = self.contour_with_max_area_ordered[2][0] - self.mean_width*2
            self.contour_with_max_area_ordered[2][1] = self.contour_with_max_area_ordered[2][1] - self.mean_width*2
            self.contour_with_max_area_ordered[3][0] = self.contour_with_max_area_ordered[3][0] - self.mean_width*2
            self.contour_with_max_area_ordered[3][1] = self.contour_with_max_area_ordered[3][1] - self.mean_width*2
            
        
        self.image_with_points_plotted = self.image.copy()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)

    def calculate_new_width_and_height_of_image(self):
        existing_image_width = self.convolved_image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        
        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def apply_perspective_transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.convolved_image, matrix, (self.new_image_width, self.new_image_height))

    def add_10_percent_padding(self):
        image_height = self.image.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def draw_contours(self):
        self.image_with_contours = self.image.copy()
        cv2.drawContours(self.image_with_contours,  [ self.contour_with_max_area ], -1, (0, 255, 0), 1)

    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis
    
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
    
    def store_process_image(self, file_name, image):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        path = os.path.join(self.output_dir, file_name)
        cv2.imwrite(path, image)
        
        