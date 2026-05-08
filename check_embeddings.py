import pickle

with open("embeddings/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["embeddings"]))
print(len(data["labels"]))
print(set(data["labels"]))