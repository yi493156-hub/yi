import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# ==============================
# Step 1: Load fer2013.csv
# ==============================
csv_path = '/Users/ly61/Desktop/fer2013.csv'  # Update with your actual csv path
df = pd.read_csv(csv_path)

# Keep only the training samples
df_train = df[df['Usage'] == 'Training'].reset_index(drop=True)
print("Original Training Samples:", len(df_train))

# Filter out invalid pixel rows (not 48x48 = 2304 pixels)
df_train = df_train[df_train['pixels'].str.split().str.len() == 2304].reset_index(drop=True)
print("Valid Training Samples:", len(df_train))

# ==============================
# Step 2: Create save paths
# ==============================
save_path = '/Users/ly61/Desktop/cnn/face'
os.makedirs(save_path, exist_ok=True)

label_csv_path = '/Users/ly61/Desktop/cnn/image_emotion.csv'
records = []  # Will store [filename, emotion]

# ==============================
# Step 3: Convert pixels → images
# ==============================
for i, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Saving images"):
    pixel_str = row['pixels']
    emotion = int(row['emotion'])

    # Convert string to 48x48 gray image
    pixels = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)

    # Save as numbered images: 0.jpg, 1.jpg, ...
    filename = f"{i}.jpg"
    cv2.imwrite(os.path.join(save_path, filename), pixels)

    # Save mapping info
    records.append([filename, emotion])

print("\nAll training images saved to:", save_path)

# ==============================
# Step 4: Save the label CSV
# ==============================
df_out = pd.DataFrame(records, columns=['image', 'emotion'])
df_out.to_csv(label_csv_path, index=False)

print("Label CSV saved to:", label_csv_path)
print("Total Labels:", len(df_out))
