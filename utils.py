import cv2

MAX_SIZE = 640


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
