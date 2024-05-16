import numpy as np
import cv2

def image_fusion_pca(img1, img2, num_components):
    # Read input images
    I1 = img1
    I2 = img2
    # Ensure both images have the same dimensions
    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # Stack pixel values of both images into a single data matrix
    data_matrix = np.vstack((I1.reshape(-1, 3), I2.reshape(-1, 3)))

    # Perform PCA
    mean = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = sorted_eigenvectors[:, :num_components]

    # Transform images to PCA feature space
    transformed_img1 = np.dot(I1.reshape(-1, 3) - mean, principal_components)
    transformed_img2 = np.dot(I2.reshape(-1, 3) - mean, principal_components)

    # Fusion rule (e.g., average)
    fused_features = (transformed_img1 + transformed_img2) / 2

    # Transform fused features back to pixel space
    fused_pixels = np.dot(fused_features, principal_components.T) + mean
    fused_image = fused_pixels.reshape(I1.shape)

    return fused_image.astype(np.uint8)




import cv2
import matplotlib.pyplot as plt

# Function to display images in a row
def plot_images_color(img1, img2, fused_img):
    plt.figure(figsize=(13,3))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1 (Original)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2 (Original)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    plt.title('Fused Image (IHS)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()





def image_fusion_ihs(img1, img2):
    # Read input images
    I1 = img1
    I2 = img2
    
    # Ensure both images have the same dimensions
    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # Convert images to the IHS color space
    I1_ihs = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    I2_ihs = cv2.cvtColor(I2, cv2.COLOR_BGR2HSV)

    # Fusion in each domain (e.g., simple averaging)
    fused_i = (I1_ihs[:,:,2].astype(float) + I2_ihs[:,:,2].astype(float)) / 2
    fused_h = (I1_ihs[:,:,0].astype(float) + I2_ihs[:,:,0].astype(float)) / 2
    fused_s = (I1_ihs[:,:,1].astype(float) + I2_ihs[:,:,1].astype(float)) / 2

    # Combine the fused components
    fused_ihs = cv2.merge((fused_h.astype(np.uint8), fused_s.astype(np.uint8), fused_i.astype(np.uint8)))

    # Convert the fused image back to the RGB color space
    fused_image = cv2.cvtColor(fused_ihs, cv2.COLOR_HSV2BGR)

    return fused_image
