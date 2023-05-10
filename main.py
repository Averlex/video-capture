import cv2 as cv
import multiprocessing as mp
import multiprocessing.connection as conn
import time
import numpy as np


"""
logs
"""


class CustomTimer:
    def __init__(self):
        """
        Initializes list of time measurements and an actual delay in ms of int type
        """
        self.values = []
        self.delay = 0

    def count(self) -> None:
        """
        Measures current time with time.perf_counter(), places value in the list and calculating actual delay in ms
        :return: None
        """
        self.values.append(time.perf_counter())
        self.delay = int((self.values[-1] - self.values[-2]) * 1000)


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

    # Adopt for possible enlarging
    def rescale_with_fields(self, width: int = -1, height: int = -1) -> None:
        if width <= 0 or height <= 0:
            pass
        else:
            act_width = self.img.shape[1]
            act_height = self.img.shape[0]
            factor = min(width / act_width, height / act_height)
            self.img = cv.resize(self.img, (int(act_width * factor), int(act_height * factor)))

            new_width = self.img.shape[1]
            new_height = self.img.shape[0]
            delta_x = int((width - new_width) / 2.)
            delta_y = int((height - new_height) / 2.)

            # !!!! Needs testing !!!!
            if delta_x > 0:
                line = np.zeros((height, delta_x, 3))
                self.img = np.hstack((line, self.img / 255, line))
            elif delta_y > 0:
                line = np.zeros((delta_y, width, 3))
                self.img = np.vstack((line, self.img / 255, line))

        return None


def get_frames(q: mp.Queue, source: int | str, external_q: mp.Queue) -> None:
    """
    Reads frames from the targeted source. Both file and camera stream are supported \n\n
    For camera: the registered delay between frames is equal to the real one or equal 1 (if there were no delay at all)\
    \n\n
    Function passes both delay and the read frame to another process for upcoming operations as a tuple\n
    Passes None as an indicator of stream ending \n
    :param q: queue used for transferring data between 2 processes
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

            # Sending for processing
            q.put((delay, frame))

            # No frame read means that no more are expected
            if not is_frame:
                break

            # Delaying the next reading (can afford since it's from the file) -> no queue overflow
            cv.waitKey(delay)
    else:
        # For time measurements
        prev = time.time()

        # Getting frames while possible
        while True:
            # Measuring time between readings and perform the next reading
            delay = int((time.time() - prev) * 1000)
            if delay == 0:
                delay = 1
            is_frame, frame = istream.read()
            prev = time.time()

            q.put((delay, frame))
            # No frame read means that no more are expected
            if is_frame is None:
                break

    # Releasing capturing object
    istream.release()

    return None


def processing(q: mp.Queue, transmitter: conn.Connection, external_q: mp.Queue) -> None:
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
        desired_width = 1024
        desired_height = 512
        # frame.rescale(640, 480)
        frame.rescale_with_fields(desired_width, desired_height)

        # Source delay is measured in msc's, that's why we use 1000. * ...
        frame.actual_delay = time.time() - prev_recording
        frame.source_fps = int(1000. / frame.delay)
        #frame.show_fps()

        # Sending forward to showing part
        transmitter.send((frame.delay, frame.img))

        # Receiving actual delay between frame readings and the frame itself
        frame.delay, frame.img = q.get()
        prev_recording = time.time()
    return None


def show_frames(receiver: conn.Connection | None, external_q: mp.Queue) -> None:
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
        cv.waitKey(1)

    return None


if __name__ == "__main__":

    # source = "World Of Warcraft - Retail 2022.01.24 - 17.54.03.01.mp4"
    source = 0
    # source = "Новиков 5.2 лекция 15.03.mkv"

    queue = mp.Queue()
    ext_q = mp.Queue()
    capture_proc = mp.Process(target=get_frames, args=(queue, source, ext_q))
    capture_proc.start()
    operation_conn, output_conn = mp.Pipe()
    operation_proc = mp.Process(target=processing, args=(queue, operation_conn, ext_q))
    operation_proc.start()
    showing_proc = mp.Process(target=show_frames, args=(output_conn, ext_q))
    showing_proc.start()

    capture_proc.join()
    operation_proc.join()
    showing_proc.join()
