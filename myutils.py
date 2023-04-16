import os
from PIL import Image, ImageOps


def load_images(directory):
    """
    Loads images from given directory. The images are resized to the specified 
    size.

    Args:
        directory (string)
        image_size (tuple)

    Returns:
        List of images (PIL Images array)
    """
    
    images = []

    filenames = os.listdir(directory)
    
    for filename in filenames:
        
        if filename.endswith(".jpg"):
                
            img = Image.open(os.path.join(directory, filename))
            images.append(img)
    
    return images


def grayscale_resize(images, image_size):
    """
    Grayscales and resizes PIL images.

    Args:
        images (PIL Images array)
    """
    gs_images = []
    
    for img in images:
        
        # grayscale image
        img = ImageOps.grayscale(img)
        
        # resize image to specified size
        img = ImageOps.fit(img, image_size, method=Image.LANCZOS)
        
        gs_images.append(img)
        
    return gs_images