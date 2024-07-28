import cv2
import numpy as np

def put_text_rectangle(
        frame: np.ndarray,  # The frame on which the text and rectangle will be drawn
        text: str,  # The text to be written on the frame
        position: tuple,  # The position of the text on the frame
        scale: int = 3,  # Scale of the text
        thickness: int = 3,  # Thickness of the text
        colour_text: tuple = (255, 255, 255),  # Colour of the text
        colour_rectangle: tuple = (255, 0, 255),  # Colour of the rectangle
        font: int = cv2.FONT_HERSHEY_PLAIN,  # Font of the text
        offset: int = 10,  # Offset from the text to the rectangle
        border: int | None = None,  # Border thickness of the rectangle
        colour_border: tuple = (0, 255, 0)  # Colour of the border
) -> tuple[np.ndarray, tuple]:
    """
    Draws a rectangle and writes text on a given frame.

    This function calculates the size of the text, draws a filled rectangle around the text, and writes the text on the frame. It also has the option to draw a border around the rectangle.

    Parameters:
    - frame: The frame on which the text and rectangle will be drawn.
    - text: The text to be written on the frame.
    - position: The position of the text on the frame.
    - scale: Scale of the text.
    - thickness: Thickness of the text.
    - colour_text: Colour of the text.
    - colour_rectangle: Colour of the rectangle.
    - font: Font of the text.
    - offset: Offset from the text to the rectangle.
    - border: Border thickness of the rectangle.
    - colour_border: Colour of the border.

    Returns:
    - frame: The frame with the drawn rectangle and text.
    - rectangle_coordinates: The coordinates of the rectangle.
    """
    # Calculate the size of the text
    (width, height), _ = cv2.getTextSize(
        text,
        font,
        scale, 
        thickness,
    )
    
    # Calculate the coordinates of the rectangle
    x_origin, y_origin = position
    x_1, y_1 = x_origin - offset, y_origin + offset  # Top-left corner
    x_2, y_2 = x_origin + width + offset, y_origin - height - offset  # Bottom-right corner

    # Draw the filled rectangle
    cv2.rectangle(
        frame,
        (x_1, y_1),
        (x_2, y_2),
        colour_rectangle,
        cv2.FILLED
    )
    
    # If border is specified, draw the border
    if border is not None:
        cv2.rectangle(
            frame,
            (x_1, y_1),
            (x_2, y_2),
            colour_border,
            border
        )
        
    # Write the text on the frame
    cv2.putText(
        frame,
        text,
        (x_origin, y_origin),
        font,
        scale,
        colour_text,
        thickness
    )
    
    # Return the frame and the coordinates of the rectangle
    return frame, (x_1, y_1, x_2, y_2)

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculates the angle between three points.

    This function calculates the angle between three points using the arctan function. It converts the points to numpy arrays for easier manipulation and handles angles greater than 180 degrees.

    Parameters:
    - a: The first point.
    - b: The second point.
    - c: The third point.

    Returns:
    - angle: The calculated angle in degrees.
    """
    # Convert points to numpy arrays for easier manipulation
    a, b, c = map(np.array, [a, b, c])

    # Calculate the angle using the arctan function
    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )
    
    # Convert radians to degrees
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Adjust the angle if it's greater than 180 degrees
    if angle > 180.0:
        angle = 360 - angle

    # Return the calculated angle
    return angle
