import cv2

MAX_SIZE = 640
DIRECTORY = "./data/labels/"

def resize_to_max(img, max_size_value=MAX_SIZE):
    max_size = max(img.shape[0], img.shape[1])
    ratio = max_size_value / max_size
    target_resolution = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    resized_image = cv2.resize(img, target_resolution)
    return resized_image


def is_image_file(filename):
    # Liste des extensions de fichiers d'image valides
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def draw_results_on_image(
    frame, results, top_left_corner, box_width, box_height, line_height
):
    # Draw a filled white rectangle
    cv2.rectangle(
        frame,
        top_left_corner,
        (top_left_corner[0] + box_width, top_left_corner[1] + box_height),
        (255, 255, 255),
        cv2.FILLED,
    )

    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 0, 0)

    # Initialize the position for the first line of text
    text_position = (top_left_corner[0] + 10, top_left_corner[1] + line_height)

    # Draw each result as a new line of text
    for label, score in results:
        text = (
            f"{label.split('.')[0]} : {score}"  # Remove the file extension from label
        )
        cv2.putText(
            frame, text, text_position, font, font_scale, text_color, font_thickness
        )
        # Move to the position of the next line
        text_position = (text_position[0], text_position[1] + line_height)
