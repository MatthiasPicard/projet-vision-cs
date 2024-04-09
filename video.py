import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from match_img import full_pipeline_real_time
import time
from utils import MAX_SIZE, DIRECTORY


def run_program():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print("picture taken, now processing...")
        # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = frame
        height, width = img_rgb.shape[:2]
        img_rgb = img_rgb[:height//2, :width//2]
        plt.imshow(img_rgb)
        plt.show()
        chosen ,_= full_pipeline_real_time(img_rgb,DIRECTORY,MAX_SIZE)
        print(f"I detected  the letter {chosen}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("waiting 5 seconds before taking another picture")
        time.sleep(5)

    cap.release()
    # cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    run_program()