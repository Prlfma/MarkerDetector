import cv2
import numpy as np

def adaptive_preprocess_image(image):
    """
    Adaptive preprocessing that applies different techniques based on image characteristics
    """
    # Always add border - this helps with markers near edges
    border_size = 1
    image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Analyze image characteristics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Apply CLAHE only if image has low contrast
    if std_intensity < 50:  # Low contrast threshold
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply bilateral filter only for noisy images (high std deviation)
    if std_intensity > 80:  # High noise threshold
        gray = cv2.bilateralFilter(gray, 5, 50, 50)  # Reduced parameters

    return gray


def multi_threshold_preprocess(image):
    """
    Try multiple preprocessing approaches and return the best one
    """
    border_size = 1
    image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing variants to try
    variants = []

    # 1. Original grayscale
    variants.append(('original', gray))

    # 2. CLAHE only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    variants.append(('clahe', clahe_applied))

    # 3. Gaussian blur for noise reduction
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(('gaussian', gaussian_blur))

    # 4. Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(('adaptive', adaptive_thresh))

    # 5. Gentle bilateral filter
    bilateral = cv2.bilateralFilter(gray, 5, 30, 30)
    variants.append(('bilateral', bilateral))

    return variants