import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from client.detector import FaceDetectorAligner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    detector = FaceDetectorAligner()

    result = detector.detect_and_align(image)

    if result is None:
        print("NO FACE DETECTED")
        return

    print("\n========== MTCNN RESULT ==========")

    print(f"Original image size:")
    print(f"  width  = {result.image_width}")
    print(f"  height = {result.image_height}")

    print("\nBounding box:")
    print(f"  x1 = {result.bbox.x1:.2f}")
    print(f"  y1 = {result.bbox.y1:.2f}")
    print(f"  x2 = {result.bbox.x2:.2f}")
    print(f"  y2 = {result.bbox.y2:.2f}")

    print("\nFace size:")
    print(f"  width  = {result.bbox.width:.2f}px")
    print(f"  height = {result.bbox.height:.2f}px")

    print("\nDetection confidence:")
    print(f"  {result.confidence:.4f}")

    print("\nFacial landmarks:")

    landmark_names = [
        "Left eye",
        "Right eye",
        "Nose",
        "Left mouth",
        "Right mouth",
    ]

    landmark_array = result.landmarks.as_array()

    for name, point in zip(landmark_names, landmark_array):
        print(f"  {name:12s}: ({point[0]:.2f}, {point[1]:.2f})")

    print("\nAligned face:")
    print(f"  shape = {result.aligned_face.shape}")
    print(f"  dtype = {result.aligned_face.dtype}")

    # ---------------------------------------------------------
    # Create visualization
    # ---------------------------------------------------------

    debug_dir = "data/debug"
    os.makedirs(debug_dir, exist_ok=True)

    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)

    x1 = result.bbox.x1
    y1 = result.bbox.y1
    x2 = result.bbox.x2
    y2 = result.bbox.y2

    # Bounding box
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=5,
    )

    # Landmarks
    radius = 7

    for point in landmark_array:
        x = float(point[0])
        y = float(point[1])

        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ],
            fill="blue",
        )

    # Save annotated original
    output_original = os.path.join(
        debug_dir,
        "mtcnn_detection.jpg",
    )

    debug_image.save(output_original)

    # Save aligned face
    output_aligned = os.path.join(
        debug_dir,
        "aligned_face.jpg",
    )

    Image.fromarray(result.aligned_face).save(
        output_aligned
    )

    print("\n========== OUTPUT FILES ==========")
    print(f"Detection visualization:")
    print(f"  {output_original}")

    print(f"\nAligned face:")
    print(f"  {output_aligned}")

    print("\nMTCNN validation completed.")


if __name__ == "__main__":
    main()
