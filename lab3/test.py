import cv2
import numpy as np

class ObjectTracker(object):
    def __init__(self, scaling_factor=1.5):
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()
        self.scaling_factor = scaling_factor
        self.frame = cv2.resize(self.frame, None,
                                fx=self.scaling_factor, fy=self.scaling_factor,
                                interpolation=cv2.INTER_AREA)

        cv2.namedWindow('Object Tracker')
        cv2.setMouseCallback('Object Tracker', self.mouse_event)

        self.selections = []
        self.drag_start = None

    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

        if self.drag_start:
            if event == cv2.EVENT_MOUSEMOVE:
                x1, y1 = x, y

                frame_copy = self.frame.copy()
                cv2.rectangle(frame_copy, self.drag_start, (x1, y1), (0, 255, 0), 2)
                cv2.imshow('Object Tracker', frame_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                x0, y0 = np.minimum(self.drag_start, (x, y))
                x1, y1 = np.maximum(self.drag_start, (x, y))

                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selections.append((x0, y0, x1, y1))

                self.drag_start = None

    def start_tracking(self):
        while True:
            _, self.frame = self.cap.read()

            self.frame = cv2.resize(self.frame, None,
                                    fx=self.scaling_factor, fy=self.scaling_factor,
                                    interpolation=cv2.INTER_AREA)

            vis = self.frame.copy()

            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, np.array((0., 60., 32.)),
                               np.array((180., 255., 255.)))

            for selection in self.selections:
                x0, y0, x1, y1 = selection
                track_window = (x0, y0, x1 - x0, y1 - y0)

                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])

                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                hist = hist.reshape(-1)

                hsv_backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
                hsv_backproj &= mask

                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, track_window = cv2.CamShift(hsv_backproj, track_window, term_crit)

                cv2.ellipse(vis, track_box, (0, 255, 0), 2)

            cv2.imshow('Object Tracker', vis)

            c = cv2.waitKey(5)
            if c == 27:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    ObjectTracker().start_tracking()
