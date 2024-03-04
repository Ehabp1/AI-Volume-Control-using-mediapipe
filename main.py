import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

vol_range = volume.GetVolumeRange()
minvol = vol_range[0]
maxvol = vol_range[1]
# volume.SetMasterVolumeLevel(-20.0, None)

video = cv.VideoCapture(0)
media = mp.solutions.hands
hand = media.Hands()
draw = mp.solutions.drawing_utils

while True:
    success, frame = video.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hand.process(rgb_frame)
    loc = []
    if result.multi_hand_landmarks:
        for det in result.multi_hand_landmarks:
            for id, lm in enumerate(det.landmark):
                h, w, c = rgb_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                loc.append([id,cx,cy])

            draw.draw_landmarks(rgb_frame, det, media.HAND_CONNECTIONS)

    if len(loc) !=0:
        cv.circle(rgb_frame,(loc[4][1],loc[4][2]),15,(255,0,255),cv.FILLED)
        cv.circle(rgb_frame, (loc[8][1], loc[8][2]), 15, (255, 0, 255), cv.FILLED)
        cv.line(rgb_frame,(loc[4][1],loc[4][2]) ,(loc[8][1], loc[8][2]),(255,0,0), 3)
        cv.circle(rgb_frame, ((loc[4][1]+loc[8][1])//2,(loc[4][2]+loc[8][2])//2), 15, (255, 0, 255), cv.FILLED)

        length = math.hypot((loc[8][1]-loc[4][1]),(loc[8][2]-loc[4][2]))
        print(length)
        vol =  np.interp(length,[50,230],(minvol,maxvol))
        volume.SetMasterVolumeLevel(vol, None)
        if length <50:
            cv.circle(rgb_frame, ((loc[4][1] + loc[8][1]) // 2, (loc[4][2] + loc[8][2]) // 2), 15, (0, 0, 255),
                      cv.FILLED)

    cv.imshow('image', rgb_frame)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()