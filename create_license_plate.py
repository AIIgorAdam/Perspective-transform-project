# create_license_plate.py

import random
from PIL import Image, ImageDraw, ImageFont

def generate_plate_number():
    """
    Generates a random plate number.
    
    Returns:
        str: The generated plate number in the format "XX-XXX-XX".
    """
    return f"{random.randint(10, 99)}-{random.randint(100, 999)}-{random.randint(10, 99)}"

def create_license_plate():
    """
   Creates a license plate image with a random number.
    
    Returns:
        tuple: A tuple containing the license plate image and a list of corner coordinates.
            - image (PIL.Image): The generated license plate image.
            - corners (list of tuples): The corner coordinates of the license plate [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    plate_number = generate_plate_number()
    width, height = 400, 100
    background_color = (255, 203, 9)
    text_color = (0, 0, 0)
    text_size = 82

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
