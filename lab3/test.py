import cv2
import numpy as np

class ObjectTracker(object):
    def __init__(self, scaling_factor=1.5):
        # ... [Previous initialization code remains unchanged]
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()
        self.scaling_factor = scaling_factor
        # Initialize variables for the second ROI and tracking state
        self.selection2 = None
        self.drag_start2 = None
        self.tracking_state2 = 0

    # Define a method to track the mouse events
    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])

        # First ROI (same as before)
        # ... [Previous mouse event handling code remains unchanged]

        # Handle the right mouse button for the second ROI
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drag_start2 = (x, y)
            self.tracking_state2 = 0

        if self.drag_start2:
            if flags & cv2.EVENT_FLAG_RBUTTON:
                h, w = self.frame.shape[:2]
                xi, yi = self.drag_start2
                x0, y0 = np.maximum(0, np.minimum([xi, yi], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xi, yi], [x, y]))
                self.selection2 = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection2 = (x0, y0, x1, y1)
            else:
                self.drag_start2 = None
                if self.selection2 is not None:
                    self.tracking_state2 = 1

    # Method to start tracking the object
    def start_tracking(self):
        while True:
            _, self.frame = self.cap.read()
            self.frame = cv2.resize(self.frame, None, 
                    fx=self.scaling_factor, fy=self.scaling_factor, 
                    interpolation=cv2.INTER_AREA)

            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            # Handle the first ROI (same as before)
            # ... [Your existing code for the first ROI]

            # Handle the second ROI
            if self.selection2:
                x0, y0, x1, y1 = self.selection2
                self.track_window2 = (x0, y0, x1-x0, y1-y0)
                hsv_roi2 = hsv[y0:y1, x0:x1]
                mask_roi2 = mask[y0:y1, x0:x1]
                hist2 = cv2.calcHist([hsv_roi2], [0], mask_roi2, [16], [0, 180])
                cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)
                self.hist2 = hist2.reshape(-1)
                vis_roi2 = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi2, vis_roi2)
                vis[mask == 0] = 0

            if self.tracking_state2 == 1:
                self.selection2 = None
                hsv_backproj2 = cv2.calcBackProject([hsv], [0], self.hist2, [0, 180], 1)
                hsv_backproj2 &= mask
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box2, self.track_window2 = cv2.CamShift(hsv_backproj2, self.track_window2, term_crit)
                cv2.ellipse(vis, track_box2, (0, 0, 255), 2)  # Using blue color for the second object

            cv2.imshow('Object Tracker', vis)
            c = cv2.waitKey(5)
            if c == 27:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    ObjectTracker().start_tracking()
