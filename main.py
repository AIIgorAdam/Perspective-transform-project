# main.py

from create_license_plate import create_license_plate
from transformations import warp_image, dewarp_image, add_gaussian_noise
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Generate the license plate image and get the source points
    license_plate_image, src_points = create_license_plate()
    license_plate_image = cv2.cvtColor(np.array(license_plate_image), cv2.COLOR_RGB2BGR)
    
    # Define the parameters
    alpha = 65  # Rotation angle around the y-axis
    beta = 65   # Rotation angle around the x-axis
    f = 400  # Assumed focal length

    # Warp the image
    warped_image, dst_points = warp_image(license_plate_image, np.array(src_points), alpha, beta, f)

    # Convert the warped image from BGR to RGB for displaying with matplotlib
    warped_image_rgb = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    # Add Gaussian noise to the L channel in HSL color space of the license plate region
    noisy_warped_image_rgb = add_gaussian_noise(warped_image_rgb, dst_points)

    # Convert the noisy image back to BGR for dewarping with OpenCV
    noisy_warped_image_bgr = cv2.cvtColor(noisy_warped_image_rgb, cv2.COLOR_RGB2BGR)

    # Dewarp the noisy image
    dewarped_image = dewarp_image(noisy_warped_image_bgr, np.array(src_points), dst_points)
    dewarped_image_rgb = cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2RGB)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    # Display the original image
    original_image_rgb = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_image_rgb)
    axes[0].axis('on')
    axes[0].set_title('Original Image')

    # Display the warped image with noise
    axes[1].imshow(noisy_warped_image_rgb)
    axes[1].axis('on')
    axes[1].set_title(f'Warped Image with Noise (alpha={alpha}, beta={beta})')

    # Display the dewarped image
    axes[2].imshow(dewarped_image_rgb)
    axes[2].axis('on')
    axes[2].set_title(f'Dewarped Image (alpha={alpha}, beta={beta})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()