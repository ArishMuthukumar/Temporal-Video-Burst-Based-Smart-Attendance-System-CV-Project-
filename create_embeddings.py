import os
import cv2
import numpy as np
import pickle #Save objects (like models, data) into a file , Load them back later
from mtcnn import MTCNN
from keras_facenet import FaceNet

print("Script started...")

# Initialize
detector = MTCNN()
embedder = FaceNet()

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings/embeddings.pkl"

known_embeddings = []
known_labels = []

# Loop through dataset
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\nProcessing person: {person_name}")

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        print(f"  Processing image: {image_name}")

        img = cv2.imread(image_path)
        if img is None:
            print("  ⚠️ Image not loaded, skipping")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face
        results = detector.detect_faces(img_rgb)

        if len(results) == 0:
            print("  ⚠️ No face detected")
            continue

        x, y, w, h = results[0]['box']
        x, y = abs(x), abs(y)

        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        # Get embedding
        embedding = embedder.embeddings([face])[0]

        known_embeddings.append(embedding)
        known_labels.append(person_name)

        print("  ✅ Done")

# Save embeddings
data = {
    "embeddings": known_embeddings,
    "labels": known_labels
}

os.makedirs("embeddings", exist_ok=True)

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully!")


print(len(known_embeddings))
print(len(known_labels))