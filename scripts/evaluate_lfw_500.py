import os
import random
import numpy as np
from PIL import Image

from client.detector import FaceDetectorAligner
from client.embeddings.facenet_extractor import FaceNetExtractor
from client.embeddings.arcface_extractor import ArcFaceExtractor
from client.fusion import FeatureFusion
from client.normalization import l2_normalize
from client.encryption.ckks_client import CKKSClient
from server.matcher import EncryptedMatcher
from common.config import VERIFICATION

LFW_ROOT = "data/lfw/lfw-deepfunneled/lfw-deepfunneled"
THRESHOLD = VERIFICATION.similarity_threshold
NUM_GENUINE = 250
NUM_IMPOSTOR = 250
SEED = 42

random.seed(SEED)

print("=" * 60)
print("LFW 500-PAIR ENCRYPTED FACE VERIFICATION EVALUATION")
print("=" * 60)
print(f"LFW DATASET PATH : {LFW_ROOT}")

if not os.path.isdir(LFW_ROOT):
    raise FileNotFoundError(f"LFW dataset not found: {LFW_ROOT}")

people = []
images = {}

for person in sorted(os.listdir(LFW_ROOT)):
    person_dir = os.path.join(LFW_ROOT, person)
    if not os.path.isdir(person_dir):
        continue

    files = sorted(
        os.path.join(person_dir, f)
        for f in os.listdir(person_dir)
        if f.lower().endswith(".jpg")
    )

    if files:
        people.append(person)
        images[person] = files

all_images = sum(len(v) for v in images.values())

print(f"LFW IDENTITIES   : {len(people)}")
print(f"LFW IMAGES       : {all_images}")
print(f"GENUINE PAIRS    : {NUM_GENUINE}")
print(f"IMPOSTOR PAIRS   : {NUM_IMPOSTOR}")
print(f"TOTAL PAIRS      : {NUM_GENUINE + NUM_IMPOSTOR}")
print(f"THRESHOLD        : {THRESHOLD}")
print("=" * 60)

# Build genuine pairs from people having at least 2 images
eligible = [p for p in people if len(images[p]) >= 2]

genuine_pairs = []
while len(genuine_pairs) < NUM_GENUINE:
    person = random.choice(eligible)
    a, b = random.sample(images[person], 2)
    genuine_pairs.append((person, a, b, 1))

# Build impostor pairs from different people
impostor_pairs = []
while len(impostor_pairs) < NUM_IMPOSTOR:
    p1, p2 = random.sample(people, 2)
    a = random.choice(images[p1])
    b = random.choice(images[p2])
    impostor_pairs.append((f"{p1} vs {p2}", a, b, 0))

pairs = genuine_pairs + impostor_pairs
random.shuffle(pairs)

detector = FaceDetectorAligner()
facenet = FaceNetExtractor()
arcface = ArcFaceExtractor()
fusion = FeatureFusion()
ckks = CKKSClient()

cache = {}

def get_embedding(path):
    if path in cache:
        return cache[path]

    face = detector.detect_and_align(Image.open(path)).aligned_face
    f = facenet.extract(face)
    a = arcface.extract(face)
    fused = fusion.fuse(f, a)
    embedding = l2_normalize(fused)

    cache[path] = embedding
    return embedding

correct = 0
genuine_scores = []
impostor_scores = []

for i, (label, path1, path2, expected) in enumerate(pairs, 1):
    e1 = get_embedding(path1)
    e2 = get_embedding(path2)

    encrypted_probe = ckks.encrypt(e1)
    encrypted_template = ckks.encrypt(e2)

    encrypted_similarity = EncryptedMatcher.compute_encrypted_similarity(
        encrypted_probe,
        encrypted_template
    )

    similarity = float(ckks.decrypt_scalar(encrypted_similarity))
    predicted = 1 if similarity >= THRESHOLD else 0

    if expected == 1:
        genuine_scores.append(similarity)
    else:
        impostor_scores.append(similarity)

    if predicted == expected:
        correct += 1

    if i <= 10 or i % 25 == 0:
        result = "CORRECT" if predicted == expected else "WRONG"
        print(
            f"[{i:03d}/500] "
            f"{'GENUINE ' if expected else 'IMPOSTOR'} "
            f"{similarity:.4f} -> "
            f"{'ACCEPT' if predicted else 'REJECT'} "
            f"[{result}]"
        )

accuracy = correct / len(pairs)
far = sum(s >= THRESHOLD for s in impostor_scores) / len(impostor_scores)
frr = sum(s < THRESHOLD for s in genuine_scores) / len(genuine_scores)

open('data/lfw_scores.txt','w').write('Genuine\n'+'\n'.join(map(str,genuine_scores))+'\nImpostor\n'+'\n'.join(map(str,impostor_scores)))

print("\n" + "=" * 60)
print("FINAL LFW EVALUATION RESULTS")
print("=" * 60)
print(f"Pairs evaluated       : {len(pairs)}")
print(f"Correct               : {correct}")
print(f"Incorrect             : {len(pairs) - correct}")
print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"FAR                   : {far * 100:.2f}%")
print(f"FRR                   : {frr * 100:.2f}%")
print(f"Mean genuine score    : {np.mean(genuine_scores):.4f}")
print(f"Mean impostor score   : {np.mean(impostor_scores):.4f}")
print(f"Minimum genuine score : {np.min(genuine_scores):.4f}")
print(f"Maximum impostor score: {np.max(impostor_scores):.4f}")
print("=" * 60)
print("LFW EVALUATION COMPLETE")
print("=" * 60)
