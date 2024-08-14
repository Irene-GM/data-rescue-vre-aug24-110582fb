import cv2
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger("ManualSelect")
logger.setLevel(logging.DEBUG)
  
class ManualSelect():

    def __init__(self, image, num_points=4, max_width = 1800, max_height = 800):

        self.image = image
        self.num_points = num_points
        self.max_width = max_width
        self.max_height = max_height
        self.points = []
        self.quit = False
        self.x_scaling = 1.0
        self.y_scaling = 1.0
        
    def detect(self):

        # displaying the image 
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.im = self.__resize_with_aspect_ratio(self.image, width=self.max_width)
        if self.im.shape[0] > self.max_height:
            self.im = self.__resize_with_aspect_ratio(self.image, height=self.max_height)

        cv2.imshow('image', self.im)
    
        # setting mouse handler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback('image', self.__click_event) 
    
        # wait for a key to be pressed to exit 
        while not self.quit:

            pressed_key = cv2.waitKey(100) & 0xFF

            if pressed_key == ord("q"):
                cv2.destroyAllWindows()
                raise Exception("Stopped corner selection")

            if pressed_key == ord("s"):
                cv2.destroyAllWindows()

                return None, None
            
            if pressed_key == ord("r"):
            
                self.im = cv2.rotate(self.im, cv2.ROTATE_180)
                self.image = cv2.rotate(self.image, cv2.ROTATE_180)
                cv2.imshow('image', self.im)

        cv2.destroyAllWindows()

        if len(self.points) != self.num_points:
            raise Exception(f"Not exactly {self.num_points} points were selected")

        return self.image, np.array(self.points)

    def __resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            self.r = height / float(h)
            dim = (int(w * self.r), height)
        else:
            self.r = width / float(w)
            dim = (width, int(h * self.r))

        return cv2.resize(image, dim, interpolation=inter)

    def __click_event(self, event, x, y, flags, params): 
    
        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
    
            # displaying the coordinates 
            # on the Shell
            logger.debug(f"({x}, {y})")

            x_scaled = int(x / self.r)
            y_scaled = int(y / self.r)

            self.points.append([x_scaled, y_scaled])
            
            self.im = cv2.circle(self.im, (x, y), 3, (0, 0, 255), -1)

            cv2.imshow('image', self.im)

            if len(self.points) >= self.num_points:
                self.quit = True