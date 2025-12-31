from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from PIL import Image
import time
import base64
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
from pathlib import Path

# Device configuration
device = 'cuda'

# Load YOLO model
print('Loading YOLO model...')
som_model = get_yolo_model('weights/icon_detect/model.pt')
print('YOLO model loaded')

# Load caption model (fine-tuned blip2 or florence2)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=device
)

# Image processing configuration
image_path = 'imgs/normal_popup.png'
# image_path = 'imgs/windows_home.png'
# image_path = 'imgs/windows_multitab.png'
# image_path = 'imgs/omni3.jpg'
# image_path = 'imgs/ios.png'
# image_path = 'imgs/word.png'
# image_path = 'imgs/excel2.png'

image = Image.open(image_path)
image_rgb = image.convert('RGB')
print('image size:', image.size)

# Drawing configuration
box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_TRESHOLD = 0.05

# OCR processing
start = time.time()
ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
    image_path,
    display_img=False,
    output_bb_format='xyxy',
    goal_filtering=None,
    easyocr_args={'paragraph': False, 'text_threshold': 0.9},
    use_paddleocr=True
)
text, ocr_bbox = ocr_bbox_rslt
cur_time_ocr = time.time()
print(f"OCR time: {cur_time_ocr - start:.2f}s")

# Get SOM labeled image
dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
    image_path,
    som_model,
    BOX_TRESHOLD=BOX_TRESHOLD,
    output_coord_in_ratio=True,
    ocr_bbox=ocr_bbox,
    draw_bbox_config=draw_bbox_config,
    caption_model_processor=caption_model_processor,
    ocr_text=text,
    use_local_semantics=True,
    iou_threshold=0.7,
    scale_img=False,
    batch_size=128,
    model_type='yolo'
)
cur_time_caption = time.time()
print(f"Caption time: {cur_time_caption - cur_time_ocr:.2f}s")
print(f"Total time: {cur_time_caption - start:.2f}s")

# Save and display labeled image
labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))

# Create output directory if it doesn't exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Generate output filename based on input image
input_filename = Path(image_path).stem
output_path = output_dir / f'{input_filename}_labeled.png'
labeled_image.save(output_path)
print(f"Labeled image saved to: {output_path}")

# Display the image
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(labeled_image)
plt.show()

# Display parsed content as DataFrame
df = pd.DataFrame(parsed_content_list)
df['ID'] = range(len(df))
print(df)

# Print parsed content list
print("\nParsed content list:")
print(parsed_content_list)
