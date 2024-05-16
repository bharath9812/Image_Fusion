import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#vgg script

# Load the saved model
model = load_model("fusion_model.keras", compile=False)

def preprocess_image(img_path):
    # Load the image in grayscale
    img = image.load_img(img_path, color_mode="grayscale", target_size=(256, 256))
    # Convert the image to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = img_array / 255.0
    return img_array

def predict_fused_image(ct_img_path, mri_img_path):
    # Preprocess the CT and MRI images
    ct_img = preprocess_image(ct_img_path)
    mri_img = preprocess_image(mri_img_path)
    print(f"Preprocessed shapes: CT: {ct_img.shape}, MRI: {mri_img.shape}")
    # Predict the fused image
    fused_img = model.predict([ct_img, mri_img])
    print(f"Raw fused image shape: {fused_img.shape}")
    
    # Denormalize the image
    fused_img = fused_img * 255.0
    # Convert to uint8
    fused_img = np.clip(fused_img, 0, 255)  # Ensure values are in the correct range
    fused_img = np.uint8(fused_img[0])
    return fused_img


# # Paths to CT and MRI images
# ct_image_path = "D:/SDP/Imfusion-main/screenshots/medical1.png"
# mri_image_path = "D:/SDP/Imfusion-main/screenshots/medical2.png"

# # Predict the fused image
# fused_image = predict_fused_image(ct_image_path, mri_image_path)

# import matplotlib.pyplot as plt

# # Load and display CT image
# ct_image = image.load_img(ct_image_path, color_mode="grayscale", target_size=(256, 256))
# plt.subplot(1, 3, 1)
# plt.imshow(ct_image, cmap='gray')
# plt.title("CT scan")
# plt.axis('off')

# # Load and display MRI image
# mri_image = image.load_img(mri_image_path, color_mode="grayscale", target_size=(256, 256))
# plt.subplot(1, 3, 2)
# plt.imshow(mri_image, cmap='gray')
# plt.title("MRI scan")
# plt.axis('off')

# # Display the fused image
# plt.subplot(1, 3, 3)
# plt.imshow(fused_image, cmap='gray')
# plt.title("Fused image")
# plt.axis('off')

# plt.tight_layout()
# plt.show()

