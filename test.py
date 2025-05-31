import cv2
import numpy as np
from glob import glob
import os

from IPreprocessor import IPreprocessor
from custom_markers_finder import MarkerDetector
import time
from EnchantPreprocessor import EnchantPreprocessor

# def test_preprocessing_variants(image_file, dictionaries):
#     """
#     Test different preprocessing variants and return the best result
#     """
#     image = cv2.imread(image_file)
#     if image is None:
#         return None, 0, 'none'
#
#     variants = multi_threshold_preprocess(image)
#     best_result = None
#     best_count = 0
#     best_method = 'none'
#
#     for method_name, processed_image in variants:
#         corners, rejected, ids, _ = detect_all_markers(processed_image, dictionaries)
#         count = len(corners) if corners else 0
#
#         if count > best_count:
#             best_count = count
#             best_result = (corners, rejected, ids, processed_image)
#             best_method = method_name
#
#     return best_result, best_count, best_method


def test_single_method(dir_name, preprocessor: IPreprocessor|None = None):
    """
    Test with a single preprocessing method
    """
    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    m=MarkerDetector()
    m.load_dictionaries("custom_dictionaries.yml", dict_names)
    os.makedirs("output\\jpg_discovery", exist_ok=True)

    image_files = glob(os.path.join(dir_name, "*.jpg"))
    failed_images = {}
    total_time = 0.0
    total_preprocess_time = 0.0
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load: {image_file}")
            continue
        start_preprocess_time= time.time()
        if preprocessor:
            processed_image = preprocessor.preprocess(image)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elapsed_preproces = time.time() - start_preprocess_time
        total_preprocess_time += elapsed_preproces
        start_time = time.time()
        corners, rejected, ids, _ = m.detect_all_markers(processed_image)
        elapsed = time.time() - start_time
        total_time += elapsed

        count = len(corners) if corners else 0

        if corners:
            output = image.copy()
            if len(output.shape) == 2:  # Grayscale
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            base_name = os.path.basename(image_file)
            cv2.imwrite(f'output\\jpg_discovery\\{base_name}', output)

        if count != 6:
            failed_images[image_file] = count

    avg_time = total_time / len(image_files)
    avg_preprocess_time = total_preprocess_time / len(image_files)

    print(f'Successful {len(image_files) - len(failed_images)}/{len(image_files)}')
    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Average preprocess time: {avg_preprocess_time:.4f} seconds")

    if len(failed_images) > 0:
        print("Failed images:")
        for name, l in failed_images.items():
            print(f"  {name}: {l}/6 markers")


if __name__ == "__main__":
    # print("Testing different preprocessing approaches:")

    # print("\n1. No preprocessing (baseline):")
    # test_single_method("jpg_discovery", None)

    # print("\n2. Adaptive preprocessing:")
    # test_single_method("jpg_discovery", adaptive_preprocess_image)

    print("\n3. Enhanced preprocessing:")
    preprocessor = EnchantPreprocessor()
    test_single_method("jpg_discovery", preprocessor)

    # print("\n4. Testing all variants (best method per image):")
    # test_all_images_with_variants("jpg_discovery")

