#!/usr/bin/env python3

# import traitlets
# from traitlets.config.configurable import SingletonConfigurable
import atexit
import time

import cv2
import threading
import numpy as np


class Camera:
    def __init__(self, src=0, width=300, height=300, rotate=False, *args, **kwargs):
        # config
        self.width = width
        self.height = height
        self.rotate = rotate
        self.exec_stop = False
        self.pause_stream = False
        # Use Below in case the camera is mounted normally
        self.gst_source = src
        self.image_array = np.empty((self.height, self.width, 3), dtype=np.uint8)

        try:
            self.cap = cv2.VideoCapture(self.gst_source, cv2.CAP_GSTREAMER)
            time.sleep(3)

            re, img = self.cap.read()
            # img = cv2.resize(img, (300, 300))

            if not re:
                raise RuntimeError('camera capture error')

            self.image_array = img
            self.start()

        except Exception as e:
            self.stop()
            raise RuntimeError('could not initialize camera')

        atexit.register(self.stop)

    def capture_frames(self):
        while True:
            if self.exec_stop:
                break

            if self.pause_stream:
                while self.pause_stream:
                    self.image_array = np.empty((self.height, self.width, 3), dtype=np.uint8)
                    time.sleep(1)
                print("Resume streaming..")

            re, img = self.cap.read()
            if re:
                if self.rotate:
                    self.image_array = cv2.rotate(img, cv2.ROTATE_180)
                else:
                    self.image_array = img
            else:
                break

    def exec_rotate(self):
        self.rotate = not self.rotate

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self.gst_source, cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self.capture_frames)
            self.thread.start()

    def pause(self):
        self.pause_stream = True

    def resume(self):
        self.pause_stream = False

    def freeze(self):
        self.pause_stream = not self.pause_stream

    def stop(self):
        self.exec_stop = True
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()
        # del self.cap

    def restart(self):
        self.stop()
        self.start()
