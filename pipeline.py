import cv2 as cv2
import asyncio

from attention.get_feature_image import get_attention_feature
from aiortc.contrib.media import MediaStreamTrack, MediaStreamError


evt_loop = asyncio.Queue()

class Pipeline:
    """
  A media sink that writes audio and/or video to a file.
  Examples:
  .. code-block:: python
      # Write to a video file.
      player = MediaRecorder('/path/to/file.mp4')
      # Write to a set of images.
      player = MediaRecorder('/path/to/file-%3d.png')
  :param file: The path to a file, or a file-like object.
  :param format: The format to use, defaults to autodect.
  :param options: Additional options to pass to FFmpeg.
  """

    def __init__(self):
        self.__tracks = list()

    def add_track(self, track: MediaStreamTrack):
        self.__tracks.append(track)
        print("GOT TRACK")

    async def start(self):
        print("STARTING")
        for track in self.__tracks:
            print(track.kind)
            asyncio.ensure_future(self.__run_track(track))

    async def __run_track(self, track):
        print(f"Running track for {track}")
        frame_num = 0
        tasks = []
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError as e:
                print(e)
                return
            data = frame.to_ndarray(format="bgr24")
            if frame_num % 30 == 0:
                task = asyncio.create_task(get_attention_feature(data, num=frame_num))
                tasks.append(task)
            frame_num += 1
