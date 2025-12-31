from util.omniparser import Omniparser
from PIL import Image
import base64
import io
import time
from pathlib import Path
import pandas as pd

# Configuration for CPU-only environment
config = {
    'som_model_path': 'weights/icon_detect/model.pt',
    'caption_model_name': 'florence2',
    'caption_model_path': 'weights/icon_caption_florence',
    'BOX_TRESHOLD': 0.05
}

# Initialize Omniparser
print('Initializing Omniparser (CPU mode)...')
start_init = time.time()
parser = Omniparser(config)
print(f'Initialization time: {time.time() - start_init:.2f}s')

# Image path to analyze
image_path = 'imgs/excel.png'
# image_path = 'imgs/windows_home.png'
# image_path = 'imgs/windows_multitab.png'

# Load and encode image to base64
image = Image.open(image_path)
print(f'Image size: {image.size}')

# Convert image to base64
buffered = io.BytesIO()
image.save(buffered, format='PNG')
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Parse the image
print('Parsing image...')
start_parse = time.time()
labeled_image_base64, parsed_content_list = parser.parse(image_base64)
print(f'Parse time: {time.time() - start_parse:.2f}s')

# Save the labeled image
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

input_filename = Path(image_path).stem
output_path = output_dir / f'{input_filename}_labeled_cpu.png'

labeled_image = Image.open(io.BytesIO(base64.b64decode(labeled_image_base64)))
labeled_image.save(output_path)
print(f'Labeled image saved to: {output_path}')

# Display parsed content as DataFrame
df = pd.DataFrame(parsed_content_list)
df['ID'] = range(len(df))
print('\nParsed content:')
print(df)

# Print parsed content list
print('\nParsed content list:')
for i, item in enumerate(parsed_content_list):
    print(f'  [{i}] {item}')
