import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def generate_plate_number():
    return f"{random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(10, 99)}"

def create_license_plate():
    plate_number = generate_plate_number()
    width, height = 400, 100
    background_color = (255, 203, 9)
    text_color = (0, 0, 0)
    text_size = 85
    
    # Create the image
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Load a font
    try:
        font = ImageFont.truetype("bahnschrift.ttf", text_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text bounding box
    text_bbox = draw.textbbox((0, 0), plate_number, font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2 - text_bbox[0]
    text_y = (height - text_height) // 2 - text_bbox[1]

    # Draw the text
    draw.text((text_x, text_y), plate_number, fill=text_color, font=font)

    # Create a new image with black background
    new_width, new_height = int(width * 1.5), int(height * 2)
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Calculate the corners of the original image in the new image
    x1 = (new_width - width) // 2
    y1 = (new_height - height) // 2
    x2 = x1 + width
    y2 = y1 + height

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # Paste the original image in the center
    new_image.paste(image, (x1, y1))

    return new_image, corners

def perspective_transform(image, corners, alpha):
    src_points = np.float32(corners)
    
     # Calculate dimensions based on the corners of the license plate
    w = corners[1][0] - corners[0][0]
   
    # Convert alpha to radians
    alpha_rad = math.radians(alpha) 

    # Calculate the distance from the center to the top/bottom edge
    z = w/2 * math.sin(alpha_rad) 

    # Define the relationship between alpha and theta
    theta_rad = 0.15 * math.exp(-0.05 * abs(alpha_rad))
 
    # Calculate the horizontal and vertical coordinates of the destination points
    x_left = corners[0][0] + w/2 * (1 - math.cos(alpha_rad)) 
    x_right = corners[0][0] + w/2 * (1 + math.cos(alpha_rad)) 
    
    y1 = corners[0][1] + z * math.tan(theta_rad)
    y2 = corners[0][1] - z * math.tan(theta_rad)
    y3 = corners[2][1] + z * math.tan(theta_rad)
    y4 = corners[2][1] - z * math.tan(theta_rad) 

    # Define the destination points
    dest_points = np.float32([
        [x_left,  y1],    # Top-left
        [x_right, y2],    # Top-right
        [x_right, y3],    # Bottom-right
        [x_left,  y4]     # Bottom-left
    ])

    # Calculate the transformation matrix 
    matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    
    # Perform the warp perspective
    transformed_img = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    
    return transformed_img, dest_points

def dewarp_transform(transformed_image, dest_points, src_points, original_size):
    # Calculate the inverse transformation matrix
    matrix_inverse = cv2.getPerspectiveTransform(dest_points, src_points)
    
    # Perform the warp perspective using the inverse matrix
    dewarped_img = cv2.warpPerspective(transformed_image, matrix_inverse, (transformed_image.shape[1], transformed_image.shape[0]), flags=cv2.INTER_LANCZOS4)
    
    # Crop the image to the original size
    width, height = original_size
    x_center, y_center = transformed_image.shape[1] // 2, transformed_image.shape[0] // 2
    x1 = x_center - width // 2
    y1 = y_center - height // 2

    cropped_img = dewarped_img[y1:y1+height, x1:x1+width]

    return cropped_img

# Main code for applying transformations and plotting
plate_image, corners = create_license_plate()
img_array = np.array(plate_image)
original_size = (400, 100)  # width and height of the original license plate image
specific_angles = [0, 65, 75, 85, -65, -75, -85]

fig, axs = plt.subplots(len(specific_angles), 2, figsize=(10, 15))

for i, angle in enumerate(specific_angles):
    transformed_image, dest_points = perspective_transform(img_array, corners, angle)
    dewarped_image = dewarp_transform(transformed_image, dest_points, np.float32(corners), original_size)

    axs[i, 0].imshow(transformed_image)
    axs[i, 0].set_title(f"Warped Tilt Angle: {angle}°")

    axs[i, 1].imshow(dewarped_image)
    axs[i, 1].set_title(f"Dewarped Tilt Angle: {angle}°")

plt.tight_layout()
plt.savefig('transformed_plates.png', dpi=300, bbox_inches='tight')  # Save the figure at high 
plt.show()
