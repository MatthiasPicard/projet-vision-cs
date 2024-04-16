import os
import cv2
import numpy as np

# Global dictionary to store descriptors of label images
label_descriptors = {}


def classify(img, labels_directory, mask_1=None):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img, mask=mask_1)

    liste_len_matching = {}

    for filename in os.listdir(labels_directory):
        if filename.endswith(".jpg"):  # Assuming the image files are JPEGs
            img_path = os.path.join(labels_directory, filename)

            mask_path = os.path.join(labels_directory + "_skin_masks", filename)
            mask = cv2.imread(mask_path, 0)

            # Retrieve or compute descriptors from memory
            if filename in label_descriptors:
                descriptors_2 = label_descriptors[filename]["descriptors"]
            else:
                img_label = cv2.imread(img_path)
                img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
                # Resize mask to match image size if necessary
                if img_label.shape != img.shape:
                    img_label = cv2.resize(img_label, (img.shape[1], img.shape[0]))
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                keypoints_2, descriptors_2 = sift.detectAndCompute(img_label, mask)
                # Store the computed descriptors in the global dictionary
                label_descriptors[filename] = {"descriptors": descriptors_2}

            matches = bf.match(descriptors_1, descriptors_2)
            liste_len_matching[filename] = len(matches)

    chosen = max(liste_len_matching, key=lambda k: liste_len_matching[k])
    return chosen, liste_len_matching


from create_skin_mask import createSkinMask
from utils import MAX_SIZE, resize_to_max, is_image_file
from matplotlib import pyplot as plt


def full_pipeline_real_time(img, labels_directory, max_size_value):
    # plt.imshow(img)
    resized_image = resize_to_max(img, max_size_value)
    skin_mask = createSkinMask(
        img=resized_image, histogram_file="histogram.plk", color_space="BGR", size=32
    )
    chosen, liste_len_matching = classify(resized_image, labels_directory, skin_mask)
    return chosen, liste_len_matching


def full_pipeline(img_path, labels_directory, max_size_value):
    img = cv2.imread(img_path)
    resized_image = resize_to_max(img, max_size_value)
    skin_mask = createSkinMask(
        img=resized_image, histogram_file="histogram.plk", color_space="BGR", size=32
    )
    chosen, liste_len_matching = classify(resized_image, labels_directory, skin_mask)
    return chosen, liste_len_matching


def run_full_pipeline():
    img_folder = "./data/test_data/"
    cumulated_precision = 0
    cumulated_good_matches = 0
    dict_results = {}
    for img_path in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_path)
        if not is_image_file(img_path):
            continue
        chosen, liste_len_matching = full_pipeline(img_path, "./data/labels", MAX_SIZE)
        # Calculate a score based on how far the real label is from the chosen one
        # For example, if image name is A_real.jpg, it should be compared to A.jpg.
        # If A is the chosen label, the score is 0. If B is the chosen label, the score is 1, etc.
        sorted_results = sorted(liste_len_matching, key=lambda k: liste_len_matching[k])
        # Find the index of the real label in the sorted results, based on the filename (must contain the first part of the real label)
        real_label_index = next(
            i
            for i, s in enumerate(sorted_results)
            if img_path.split("/")[-1].split("_")[0] in s
        )
        max_matches = liste_len_matching[chosen]
        num_matches = liste_len_matching[sorted_results[real_label_index]]
        precision = num_matches / max_matches * 100

        print(
            f"Image {img_path} is most similar to {chosen} with {liste_len_matching[chosen]} matches. Real one has {num_matches} matches."
        )
        print("Precision :", precision, "%")
        cumulated_precision += precision
        if img_path.split("/")[-1].split("_")[0] == chosen.split(".")[0]:
            cumulated_good_matches += 1
        dict_results[img_path] = (chosen, liste_len_matching[chosen], precision)
    return (
        cumulated_precision / len(os.listdir(img_folder)),
        cumulated_good_matches,
        dict_results,
    )


if __name__ == "__main__":
    score, correct, results = run_full_pipeline()
    print(f"Precision dans le matching des keypoints: {score} %")
    print(f"Nombre d'images correctement identifiées: {correct}")

    print("Results:", results)
    # Plot an histogram of the score for each image label
    labels = [key.split("/")[-1].split("_")[0] for key in results]
    scores = [results[key][2] for key in results]
    plt.bar(labels, scores)
    plt.xlabel("Labels")
    plt.ylabel("Scores")
    plt.title("Scores of each label")
    plt.show()
