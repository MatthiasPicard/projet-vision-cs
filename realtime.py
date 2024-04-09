import cv2
from create_skin_mask import createSkinMask
from match_img import classify
from utils import draw_results_on_image

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Définir les coordonnées du rectangle (x, y, largeur, hauteur)
# x et y définissent le coin supérieur gauche du rectangle
x, y, w, h = 1300, 500, 600, 400

while True:
    # Capturer frame par frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Appliquer l'effet miroir à l'image
    frame = cv2.flip(frame, 1)

    # Dessiner un rectangle autour de la région d'intérêt (ROI)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extraire la région d'intérêt (ROI) de l'image
    roi = frame[y : y + h, x : x + w]

    # Re Appliquer l'effet miroir à l'image a traiter
    roi = cv2.flip(roi, 1)

    # Afficher la résolution de la frame
    print(f"Resolution: {roi.shape[1]} x {roi.shape[0]}")

    # Find the skin mask of the ROI
    skin_mask = createSkinMask(
        roi, histogram_file="histogram.plk", color_space="BGR", size=32
    )

    path_to_labels = "./data/labels"
    chosen, liste_len_matching = classify(roi, path_to_labels, skin_mask)
    sorted_results = sorted(
        liste_len_matching.items(), key=lambda x: x[1], reverse=True
    )

    # Define position and size of the results box
    results_top_left = (10, 10)
    results_box_width = 150
    results_box_height = 20 * len(sorted_results)  # 20 pixels per line of text
    results_line_height = 20

    # Draw the results on the frame
    draw_results_on_image(
        frame,
        sorted_results,
        results_top_left,
        results_box_width,
        results_box_height,
        results_line_height,
    )

    # Afficher la frame avec le rectangle
    cv2.imshow("Webcam Live", frame)

    # Afficher le masque de peau
    cv2.imshow("Skin Mask", skin_mask)

    # Appuyez sur 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Lorsque tout est fait, libérer la capture
cap.release()
cv2.destroyAllWindows()
