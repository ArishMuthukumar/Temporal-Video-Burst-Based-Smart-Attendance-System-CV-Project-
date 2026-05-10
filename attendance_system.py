import cv2
import numpy as np
import pickle
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("embeddings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_labels = np.array(data["labels"])

# Unique students
students = list(set(known_labels))

# Initialize presence table
intervals = 5
attendance = {student: [0]*intervals for student in students}

# Initialize models
detector = MTCNN()
embedder = FaceNet()

# Video
cap = cv2.VideoCapture("test_videos/Test1.mp4")

THRESHOLD = 0.6
FRAME_SKIP = 3
INTERVAL_FRAMES = 60   # adjust based on video length

frame_count = 0
interval_index = 0

# Temporary counter for interval
interval_counts = {student: 0 for student in students}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    for result in results:
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)

        face = rgb[y:y+h, x:x+w]
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])[0]

        similarities = cosine_similarity([embedding], known_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > THRESHOLD:
            name = known_labels[best_idx]
            interval_counts[name] += 1

            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Interval check
    if frame_count % INTERVAL_FRAMES == 0:
        print(f"Interval {interval_index+1} processing...")

        for student in students:
            if interval_counts[student] > 2:  # threshold
                attendance[student][interval_index] = 1

        interval_counts = {student: 0 for student in students}
        interval_index += 1

        if interval_index >= intervals:
            break

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final voting
final_result = {}
for student in students:
    if sum(attendance[student]) >= 3:
        final_result[student] = "Present"
    else:
        final_result[student] = "Absent"

# Save CSV
df = pd.DataFrame(attendance).T
df.columns = [f"I{i+1}" for i in range(intervals)]
df["Final"] = [final_result[s] for s in df.index]

df.to_csv("output/attendance.csv")

print("\nFinal Attendance:")
print(df)