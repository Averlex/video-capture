import cv2 as cv
from typing import Any, overload
import multiprocessing as mp
import multiprocessing.connection as conn
import time
import numpy as np


class FrameProcessor:
    def __init__(self, frame: np.ndarray, source_delay: int):
        self.img: np.ndarray | None = frame
        self.delay = source_delay

        if self.img is not None:
            self.width = self.img.shape[1]
            self.height = self.img.shape[0]
            self.__source_width = self.width
            self.__source_height = self.height

        self.actual_delay = 0
        self.source_fps = int(1000. / source_delay)
        self.actual_fps = 0

        return

    def calc_fps(self) -> int:
        if self.actual_delay != 0:
            # Local fps is measured in seconds
            self.actual_fps = int(1. / self.actual_delay)
        return self.actual_fps

    def show_fps(self) -> None:
        self.img = cv.putText(self.img, "{:.0f} Source FPS".format(self.source_fps),
                              (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (62, 152, 40), thickness=2)
        self.img = cv.putText(self.img, "{:.0f} Possible FPS".format(self.calc_fps()),
                              (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1.0, (249, 54, 54), thickness=2)
        return None

    def rescale(self, width: int = -1, height: int = -1, xfact: float = 1., yfact: float = 1.) -> None:
        if width > 0 and height > 0:
            self.img = cv.resize(self.img, (width, height))
        else:
            self.img = cv.resize(self.img, (int(self.width * xfact), int(self.height * yfact)))
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        return None


def get_frames(q: mp.Queue, source: int | str) -> None:
    """
    Reads frames from the targeted source. Both file and camera stream are supported \n\n
    For camera: the registered delay between frames is equal to the real one or equal 1 (if there were no delay at all)\
    \n\n
    Function passes both delay and the read frame to another process for upcoming operations as a tuple\n
    Passes None as an indicator of stream ending \n
    :param q: queue used for trasfering data between 2 processes
    :param source: video source. Expected 0 for camera or a string value for a file
    :return: None
    :raises Nothing: TODO
    """
    # Initializing capture object
    istream = cv.VideoCapture(source)
    is_frame = True
    delay = 0          # Actual delay
    fps = 0            # For video capture (not cam)

    if type(source) == str:
        # Getting fps and calculation the delay in msecs
        fps = istream.get(cv.CAP_PROP_FPS)
        delay = int(1000. / fps)

        # Getting frames while possible
        while True:
            is_frame, frame = istream.read()
            # No frame read means that no more are expected
            if not is_frame:
                q.put((delay, None))
                break

            # Sending for processing
            q.put((delay, frame))

            # Delaying the next reading (can afford since it's from the file) -> no queue overflow
            cv.waitKey(delay)
    else:
        # For time measurements
        prev = time.time()

        # Getting frames while possible
        while True:
            # Measuring time between readings and perform the next reading
            delay = int(time.time() - prev)
            if delay == 0:
                delay = 1
            is_frame, frame = istream.read()
            prev = time.time()

            # No frame read means that no more are expected
            if not is_frame:
                q.put((delay, None))
                break
            else:
                q.put((delay, frame))

    # Releasing capturing object
    istream.release()

    return None


def processing(q: mp.Queue, transmitter: conn.Connection) -> None:
    """
    Processes the incoming frame and then transferring it to another process responsible for drawing
    :param q: queue used for transferring source frames to this process. Tuples of (delay, frame) are expected
    :param transmitter: connection object used for transferring same tuple for drawing after processing
    :return: None
    :raises Nothing: TODO
    """
    # Getting the first frame outside a cycle to grab source data
    tmp_delay, tmp_frame = q.get()
    frame = FrameProcessor(tmp_frame, tmp_delay)

    prev_recording = time.time()
    # Processing incoming frames till there are no frames left
    while True:

        # None means no more frames left
        if frame.img is None:
            transmitter.send((frame.delay, frame.img))
            break

        # Do some stuff here
        frame.rescale(640, 480)

        # Source delay is measured in msc's, that's why we use 1000. * ...
        frame.actual_delay = time.time() - prev_recording
        frame.source_fps = int(1000. / frame.delay)
        frame.show_fps()

        # Sending forward to showing part
        transmitter.send((frame.delay, frame.img))

        # Receiving actual delay between frame readings and the frame itself
        frame.delay, frame.img = q.get()
        prev_recording = time.time()
    return None


def show_frames(receiver: conn.Connection | None) -> None:
    """
    Function designed for raw frame drawing in a separate process with a given delay
    :param receiver: connection object used for receiving data. Tuples (delay, frame) expected
    :return: None
    :raises Nothing: TODO
    """
    # Showing frames while there are frames to show
    while True:
        # Receiving actual delay between frame readings
        delay, frame = receiver.recv()
        # Destroying windows, closing pipes
        if frame is None:
            cv.destroyAllWindows()
            receiver.close()
            break

        cv.imshow("Some video", frame)
        cv.waitKey(delay)

    return None


if __name__ == "__main__":

    source = "World Of Warcraft - Retail 2022.01.24 - 17.54.03.01.mp4"
    # source = 0
    # source = "Новиков 5.2 лекция 15.03.mkv"

    queue = mp.Queue()
    capture_proc = mp.Process(target=get_frames, args=(queue, source,))
    capture_proc.start()
    operation_conn, output_conn = mp.Pipe()
    operation_proc = mp.Process(target=processing, args=(queue, operation_conn,))
    operation_proc.start()
    showing_proc = mp.Process(target=show_frames, args=(output_conn,))
    showing_proc.start()

    capture_proc.join()
    operation_proc.join()
    showing_proc.join()
