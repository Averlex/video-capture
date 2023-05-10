import cv2 as cv
import multiprocessing as mp
import multiprocessing.connection as conn
import time
import numpy as np
from datetime import datetime

"""
TODO: improve logs, write deltas
TODO: calculating resizing params on class init for FrameProcessor (width/height ratios)

NOTE: All measurements are performed AFTER reading/getting/showing new frame
"""


class CustomTimer:
    """
    A simple timer intended for storing time measurements and calculating current delay
    """
    def __init__(self):
        """
        Initializes list of time measurements and an actual delay in ms of int type
        """
        self.values = [time.time()]
        self.delay = 1

    def count(self) -> None:
        """
        Measures current time with time.perf_counter(), places value in the list and calculating actual delay in ms
        :return: None
        """
        self.values.append(time.time())
        self.delay = int((self.values[-1] - self.values[-2]) * 1000)


class FrameProcessor:
    """
    Class that working with a single frame
    """
    def __init__(self, frame: np.ndarray | None):
        """
        On init defines the shape of the frame
        :param frame: source <ndarray> storing the frame data
        """
        self.img: np.ndarray | None = frame
        # Actual delay measured from the source
        self.delay = 1

        # Getting the frame shape, None in case no frames were read
        if self.img is not None:
            self.width = self.img.shape[1]
            self.height = self.img.shape[0]
            self.__source_width = self.width
            self.__source_height = self.height

        # Actual delay (the one we get during processing)
        self.actual_delay = 1

        return

    def show_fps(self) -> None:
        """
        Method shows fps values on the current frame: source(expected) and the real one (got from processing) \n
        Both values are calculated based on delay values stored
        :return: None
        """
        source_fps = int(1000. / self.delay)
        actual_fps = int(1000. / self.actual_delay)

        self.img = cv.putText(self.img, "{:.0f} Source FPS".format(source_fps),
                              (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (62, 152, 40), thickness=2)
        self.img = cv.putText(self.img, "{:.0f} Output FPS".format(actual_fps),
                              (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1.0, (249, 54, 54), thickness=2)
        return None

    def rescale(self, width: int = -1, height: int = -1, xfact: float = 1., yfact: float = 1.) -> None:
        """
        Deprecated; used for simple rescaling. Might be needed later, but I'm not yet sure about it
        :param width:
        :param height:
        :param xfact:
        :param yfact:
        :return:
        """
        if width > 0 and height > 0:
            self.img = cv.resize(self.img, (width, height))
        else:
            self.img = cv.resize(self.img, (int(self.width * xfact), int(self.height * yfact)))
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        return None

    def rescale_with_fields(self, width: int = -1, height: int = -1) -> None:
        """
        Rescales the frame to the set values. Add black lines if ratio differs \n
        TODO: adopt for possible frame enlarging. Need some overall testing
        :param width: width to set
        :param height: height to set
        :return: None
        """
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
    :param external_q: mp.Queue object used to deliver timings to the host process
    :return: None
    """
    # Initializing capture object
    istream = cv.VideoCapture(source)
    is_frame = True
    delay = 0                       # Actual delay
    fps = 0                         # For video capture (not cam)
    timings = CustomTimer()         # For time measurements

    if type(source) == str:
        # Getting fps and calculation the delay in msecs
        fps = istream.get(cv.CAP_PROP_FPS)
        delay = 1000. / fps         # For passing to other workers
        int_delay = int(delay)      # For delaying reading

        # Getting frames while possible
        while True:
            is_frame, frame = istream.read()
            timings.count()

            # Sending for processing
            q.put((timings.delay, frame))

            # No frame read means that no more are expected
            if not is_frame:
                break

            # Delaying the next reading (can afford since it's from the file) -> no queue overflow
            cv.waitKey(int_delay)
    else:
        # Getting frames while possible
        while True:
            # Measuring time between readings and perform the next reading
            is_frame, frame = istream.read()
            timings.count()

            q.put((timings.delay, frame))
            # No frame read means that no more are expected
            if is_frame is None:
                break

    # Releasing capturing object
    istream.release()
    external_q.put(timings)

    return None


def processing(q: mp.Queue, transmitter: conn.Connection, external_q: mp.Queue) -> None:
    """
    Processes the incoming frame and then transferring it to another process responsible for drawing
    :param q: queue used for transferring source frames to this process. Tuples of (delay, frame) are expected
    :param transmitter: connection object used for transferring same tuple for drawing after processing
    :param external_q: mp.Queue object used to deliver timings to the host process
    :return: None
    """
    timings = CustomTimer()
    # Getting the first frame outside a cycle to grab source data
    tmp_delay, tmp_frame = q.get()
    timings.count()
    frame = FrameProcessor(tmp_frame)
    frame.delay = tmp_delay
    frame.actual_delay = timings.delay

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
        frame.show_fps()

        # Sending forward to showing part
        transmitter.send((frame.delay, frame.img))

        # Receiving actual delay between frame readings and the frame itself
        frame.delay, frame.img = q.get()
        timings.count()
        frame.actual_delay = timings.delay

    external_q.put(timings)

    return None


def show_frames(receiver: conn.Connection | None, external_q: mp.Queue) -> None:
    """
    Function designed for raw frame drawing in a separate process with a given delay
    :param receiver: connection object used for receiving data. Tuples (delay, frame) expected
    :param external_q: mp.Queue object used to deliver timings to the host process
    :return: None
    """
    timings = CustomTimer()
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
        timings.count()
        cv.waitKey(1)

    external_q.put(timings)

    return None


if __name__ == "__main__":

    source = "World Of Warcraft - Retail 2022.01.24 - 17.54.03.01.mp4"
    # source = 0
    # source = "Новиков 5.2 лекция 15.03.mkv"

    # Internal queue for process data exchanging (reading and processing)
    inner_queue = mp.Queue()
    # External queues for collecting time measurements from workers
    reader_q, proc_q, show_q = mp.Queue(), mp.Queue(), mp.Queue()
    # Connectors for exchanging data between processing and output workers
    operation_conn, output_conn = mp.Pipe()

    # Reading process init
    capture_proc = mp.Process(target=get_frames, args=(inner_queue, source, reader_q))
    capture_proc.start()

    # Processing init
    operation_proc = mp.Process(target=processing, args=(inner_queue, operation_conn, proc_q))
    operation_proc.start()

    # Output init
    showing_proc = mp.Process(target=show_frames, args=(output_conn, show_q))
    showing_proc.start()

    # Collecting time measurements from processes
    times = [reader_q.get(), proc_q.get(), show_q.get()]

    # Locking until each process is done
    capture_proc.join()
    operation_proc.join()
    showing_proc.join()

    # Some simple logging
    with open("time_logs.txt", 'w') as f:
        def normalize(val: float) -> str: return datetime.fromtimestamp(val).strftime("%H:%M:%S.%f")
        f.write(f"Reading init: {normalize(times[0].values[0])},\tprocessing init: {normalize(times[1].values[0])},\t"
                f"output init: {normalize(times[2].values[0])}\n")
        f.write("-" * 100 + "\n")
        f.write("Frame\t|\tRead\t\t|\tProcessed\t|\tShown\n")
        f.write("-" * 80 + "\n")
        for indx in range(1, len(times[2].values)):
            f.write(f"{indx}\t|\t{normalize(times[0].values[indx])}\t|\t"
                    f"{normalize(times[1].values[indx])}\t|\t"
                    f"{normalize(times[2].values[indx])}\n")
