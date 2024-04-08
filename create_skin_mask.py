import cv2
import numpy as np
import os
import pickle


def compterPixels(I, k):
    return np.count_nonzero((I == k))


def inverserImage(I):
    return 255 - I


def getImagesFromResources(skin_rep, color_space):
    skindir = os.path.dirname(skin_rep)
    print(skindir)
    skin_images = []
    for filename in os.listdir(skindir):
        img = cv2.imread(os.path.join(skindir, filename))
        if img is not None:
            if color_space == "BGR":
                skin_images.append(img)
            elif color_space == "Lab":
                skin_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            elif color_space == "HSV":
                skin_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            elif color_space == "Gray":
                skin_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return skin_images


def create_histograms(color_space, size, skin_images, mask_images):
    if color_space in ["BGR", "Lab"]:
        # Creation de l'histogramme de peau
        hist_peau = [[0 for j in range(size)] for i in range(size)]
        hist_non_peau = [[0 for j in range(size)] for i in range(size)]
        nb_pixel_peau = 0
        nb_pixel_non_peau = 0
        for img in skin_images:
            nb_pixel_peau = nb_pixel_peau + compterPixels(mask_images[0], 255)
            nb_pixel_non_peau = nb_pixel_non_peau + compterPixels(mask_images[0], 0)
            hist_peau = hist_peau + cv2.calcHist(
                [skin_images[0]], [1, 2], mask_images[0], [size, size], [0, 256, 0, 256]
            )
            hist_non_peau = hist_non_peau + cv2.calcHist(
                [skin_images[0]],
                [1, 2],
                inverserImage(mask_images[0]),
                [size, size],
                [0, 256, 0, 256],
            )
        print(nb_pixel_peau)
        hist_peau = hist_peau / nb_pixel_peau
        hist_non_peau = hist_non_peau / nb_pixel_non_peau
        p_peau_prior = nb_pixel_peau / (nb_pixel_peau + nb_pixel_non_peau)
        p_non_peau_prior = nb_pixel_non_peau / (nb_pixel_peau + nb_pixel_non_peau)
    elif color_space == "HSV":
        hist_peau = [[0 for j in range(size)] for i in range(size)]
        hist_non_peau = [[0 for j in range(size)] for i in range(size)]
        nb_pixel_peau = 0
        nb_pixel_non_peau = 0
        for img in skin_images:
            nb_pixel_peau = nb_pixel_peau + compterPixels(mask_images[0], 255)
            nb_pixel_non_peau = nb_pixel_non_peau + compterPixels(mask_images[0], 0)
            hist_peau = hist_peau + cv2.calcHist(
                [skin_images[0]], [0, 1], mask_images[0], [size, size], [0, 180, 0, 256]
            )
            hist_non_peau = hist_non_peau + cv2.calcHist(
                [skin_images[0]],
                [0, 1],
                inverserImage(mask_images[0]),
                [size, size],
                [0, 180, 0, 256],
            )
        print(nb_pixel_peau)
        hist_peau = hist_peau / nb_pixel_peau
        hist_non_peau = hist_non_peau / nb_pixel_non_peau
        p_peau_prior = nb_pixel_peau / (nb_pixel_peau + nb_pixel_non_peau)
        p_non_peau_prior = nb_pixel_non_peau / (nb_pixel_peau + nb_pixel_non_peau)
    return hist_peau, hist_non_peau, p_peau_prior, p_non_peau_prior


def load_or_create_histograms(histogram_file, color_space, size, skin_dir, mask_dir):
    if os.path.exists(histogram_file):
        with open(histogram_file, "rb") as file:
            hist_peau, hist_non_peau = pickle.load(file)
    else:
        skin_images = getImagesFromResources(skin_dir, color_space)
        mask_images = getImagesFromResources(mask_dir, "Gray")
        hist_peau, hist_non_peau, _, _ = create_histograms(
            color_space, size, skin_images, mask_images
        )
        with open(histogram_file, "wb") as file:
            pickle.dump((hist_peau, hist_non_peau), file)
    return hist_peau, hist_non_peau


def is_skin_from_histo(pixel, histo_peau, histo_non_peau, size):
    c1, c2 = int(pixel[1] * (size - 1) / 255), int(pixel[2] * (size - 1) / 255)
    return histo_peau[c1, c2] > histo_non_peau[c1, c2]


def createSkinMaskPath(image_path, histogram_file, color_space, size):
    hist_peau, hist_non_peau = load_or_create_histograms(
        histogram_file,
        color_space,
        size,
        "./data/Data/Face_Dataset/Pratheepan_Dataset/FacePhoto/",
        "./data/Data/Face_Dataset/Ground_Truth/GroundT_FacePhoto/",
    )
    img = cv2.imread(image_path)
    if color_space == "BGR":
        img = img
    elif color_space == "Lab":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if is_skin_from_histo(img[i, j], hist_peau, hist_non_peau, size):
                mask[i, j] = 255
    return mask


def createSkinMask(img, histogram_file, color_space, size):
    hist_peau, hist_non_peau = load_or_create_histograms(
        histogram_file,
        color_space,
        size,
        "./data/Data/Face_Dataset/Pratheepan_Dataset/FacePhoto/",
        "./data/Data/Face_Dataset/Ground_Truth/GroundT_FacePhoto/",
    )
    if color_space == "BGR":
        img = img
    elif color_space == "Lab":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if is_skin_from_histo(img[i, j], hist_peau, hist_non_peau, size):
                mask[i, j] = 255
    return mask


# # Example usage
# # mask = createSkinMaskPath("./data/A_real.jpg", "histograms.pkl", "Lab", 32)
# img_path = "./data/A_real.jpg"
# img = cv2.imread(img_path)
# mask = createSkinMask(img, "histograms.pkl", "Lab", 32)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
