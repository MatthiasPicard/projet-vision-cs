import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time



def run_program():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # plt.imshow(img_rgb)
        # plt.show() 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(10)

    cap.release()
    # cv2.destroyAllWindows()
    
    
    if __name__ == "__main__":
        run_program()