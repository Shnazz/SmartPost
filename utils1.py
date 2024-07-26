import cv2
import numpy as np

def put_text_rectangle(
        frame,
        text,
        position,
        scale=3,
        thickness=3,
        colour_text=(255, 255, 255),
        colour_rectangle=(255, 0, 255),
        font=cv2.FONT_HERSHEY_PLAIN,
        offset=10,
        border=None,
        colour_border=(0, 255, 0)
):
    (width, height), _ = cv2.getTextSize(
        text,
        font,
        scale, 
        thickness,
    )
    
    x_origin, y_origin = position
    x_1 = x_origin - offset
    y_1 = y_origin + offset
    x_2 = x_origin + width + offset
    y_2 = y_origin - height - offset

    cv2.rectangle(
        frame,
        (x_1, y_1),
        (x_2, y_2),
        colour_rectangle,
        cv2.FILLED
    )
    
    if border is not None:
        cv2.rectangle(
            frame,
            (x_1, y_1),
            (x_2, y_2),
            colour_border,
            border
        )
        
    cv2.putText(
        frame,
        text,
        (x_origin, y_origin),
        font,
        scale,
        colour_text,
        thickness
    )
    
    return frame, (x_1, y_1, x_2, y_2)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )
    
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle
