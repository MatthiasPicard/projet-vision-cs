import os
import cv2


def classify(img, labels_directory, mask_1=None):

    liste_len_matching = {}
    sift = cv2.SIFT_create()  # we could also use ORB but I dont think it work better
    bf = cv2.BFMatcher(crossCheck=True)  # we can specify another norm than L2

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img, mask=mask_1)

    # height, width = img.shape[:2]

    for filename in os.listdir(labels_directory):
        img_path = os.path.join(labels_directory, filename)
        img_label = cv2.imread(img_path)
        img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(labels_directory + "_skin_masks/" + filename, 0)

        # If the label image and/or its mask are not the same size than the image, we resize them
        if img_label.shape != img.shape:
            # img_label_resized = cv2.resize(img_label, (width, height))
            img_label = cv2.resize(img_label, (img.shape[1], img.shape[0]))
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        keypoints_2, descriptors_2 = sift.detectAndCompute(img_label, mask=mask)
        matches = bf.match(descriptors_1, descriptors_2)
        liste_len_matching[filename] = len(matches)

        matches = sorted(matches, key=lambda x: x.distance)

        # # Draw the top 50 matches on the resized label image
        # img_matches = cv2.drawMatches(
        #     img,
        #     keypoints_1,
        #     img_label,
        #     keypoints_2,
        #     matches[:50],
        #     None,
        #     flags=2,
        # )

        # # Display the matching keypoints
        # plt.imshow(img_matches), plt.show()

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
    return cumulated_precision / len(os.listdir(img_folder))


if __name__ == "__main__":
    score = run_full_pipeline()
    print(f"Precision dans le matching des keypoints: {score} %")
