#   this code ROTATES (and ZOOMS) the video relative to the mouse position
    #   it also creates a "box" around the mouse to signal the position of the oft / experiment border location.

import cv2
from utilities import terminal_colors as colors
from tqdm import tqdm
import pandas as pd
import numpy as np
import math

class VideoRotator:
    
    def __init__(self, video_path : str, output_path : str, out_width : int, out_height : int):
        """ 
        Docstring for __init__
        
        :param self: Description
        :param video_path: Description
        :type video_path: str
        :param output_path: Description
        :type output_path: str
        :param zoom: how much you want to zoom, None = 1; 5 means a X5 zoom.
        :type zoom: float
        """

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise KeyError(f"{video_path} is not a valid video path")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height), isColor=False)
        if not self.out.isOpened():
            raise KeyError(f"{output_path} is not a valid output path")

        self.crop_x0 = (self.original_width - out_width) // 2
        self.crop_y0 = (self.original_height - out_height) // 2
        self.crop_x1 = self.crop_x0 + out_width
        self.crop_y1 = self.crop_y0 + out_height


    def follow(self, tracking_dataframe : pd.DataFrame, centre : str, endpoint : str):

        with tqdm(desc = colors.GREEN +"rotating" + colors.ENDC, total = self.total_frames, ascii = True) as pbar:
            for n in range(0, self.total_frames):
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    gray_frame = np.zeros(
                        (self.original_height, self.original_width),
                        dtype=np.uint8
                    )
                    print(colors.FAIL + f"frame {n} not read was imputed" + colors.ENDC)

                px = tracking_dataframe.loc[n, endpoint+".x"]
                py = tracking_dataframe.loc[n, endpoint+".y"]
                cx = tracking_dataframe.loc[n, centre+".x"]
                cy = tracking_dataframe.loc[n, centre+".y"]

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                tx = -(cx - self.original_width/2)
                ty = -(cy - self.original_height/2)
                traslation_matrix = np.float32([
                                                 [1, 0, tx],
                                                 [0, 1, ty]
                                                 ])
                traslated_frame = cv2.warpAffine(gray_frame, traslation_matrix,(self.original_width, self.original_height))

                angle = math.degrees(math.atan2(py - cy, px - cx)) + 90
                rotation_matrix = cv2.getRotationMatrix2D((self.original_width / 2, self.original_height / 2), angle, 1 )
                rotated_frame = cv2.warpAffine(traslated_frame, rotation_matrix, (self.original_width, self.original_height))

                cropped_frame = rotated_frame[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]

                normalized_frame = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                self.out.write(normalized_frame)
                pbar.update(1)
        self.close()

    def close(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

video_names = []

video_names.append("3278_21min_behaviour_2023-01-19T11_08_30.mp4")
video_names.append("3279_21min_behaviour_2023-01-19T12_57_29.mp4")
video_names.append("BehavioralCamera2023-03-09T10_37_32.mp4")
video_names.append("BehavioralCamera2023-03-09T11_04_40.mp4")
video_names.append("BehavioralCamera2023-03-09T11_41_07.mp4")
video_names.append("BehavioralCamera2023-03-09T12_34_50.mp4")

video_names.append("MBT1-M2.mp4")
video_names.append("MBT1-M3.mp4")
video_names.append("MBT1-M6.mp4")
video_names.append("MBT1-M7.mp4")
video_names.append("MBT1-M10.mp4")
video_names.append("MBT1-M11.mp4")
video_names.append("MBT1-M14.mp4")
video_names.append("MBT1-M15.mp4")
video_names.append("MBT1-M18.mp4")

for v in video_names:
    rotator = VideoRotator(f"./data/raw_videos/{v}", f"./data/rotated_videos/{v[:-4]}" + ".mp4", 76, 142)
    rotator.follow(pd.read_csv(f"./data/features/{v[:-4]}" + ".csv"), "mouse_top.mouse_top_0.bodycentre","mouse_top.mouse_top_0.neck")
