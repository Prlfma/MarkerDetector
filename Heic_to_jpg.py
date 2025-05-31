from PIL import Image
import pillow_heif


def convert_heic_to_jpg(input_path, output_path=None):
    pillow_heif.register_heif_opener()

    image = Image.open(input_path)

    if output_path is None:
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = 'jpg_discovery\\' + name + ".jpg"

    image.save(output_path, "JPEG")
    print(f"Конвертовано: {output_path}")


import glob
import os

heic_files = glob.glob(os.path.join("D:\TimeAndSpace\discovery","*.heic"))

for heic in heic_files:
    convert_heic_to_jpg(heic)

