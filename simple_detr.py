from rfdetr.detr import RFDETRMedium
import cv2
import numpy as np
from pathlib import Path

# Load the model with your trained weights
model = RFDETRMedium(pretrain_weights="./weights/ui_detr/model.pth", resolution=1600)

# Process an image
image_path = "./imgs/screenshot.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
detections = model.predict(image_rgb, threshold=0.3)

# Get results
boxes = detections.xyxy  # Bounding boxes
scores = detections.confidence  # Confidence scores

print(f"Detected {len(boxes)} UI elements")

# Draw bounding boxes on image
result_image = image.copy()
for i, (box, score) in enumerate(zip(boxes, scores)):
    x1, y1, x2, y2 = map(int, box)
    # Draw rectangle
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw label with ID and confidence
    label = f"{i}: {score:.2f}"
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
    cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Save result image
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
input_filename = Path(image_path).stem
output_path = output_dir / f"{input_filename}_detr_result.png"
cv2.imwrite(str(output_path), result_image)
print(f"Result saved to: {output_path}")