from PIL import Image
from PIL.ExifTags import TAGS
import os
from datetime import datetime


def get_exif_data(image_path: str) -> dict:
    image = Image.open(image_path)
    exif_data = image._getexif()
    exif_dict = {}
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            exif_dict[tag_name] = value 
    return exif_dict


def calculate_minutes_difference(date1_str: str, date2_str: str) -> float:
    date_format = "%Y:%m:%d %H:%M:%S"
    date1 = datetime.strptime(date1_str, date_format)
    date2 = datetime.strptime(date2_str, date_format)
    time_difference = date2 - date1
    difference_in_minutes = time_difference.total_seconds() / 60
    return difference_in_minutes


project_root = os.path.abspath(os.path.dirname(__file__))
project_root = project_root.replace('\\registration', '')
data_path = os.path.join(project_root, 'data', 'traps', '1', 'IMG_0004.JPG')
exif_data = get_exif_data(data_path)
date1_str = exif_data['DateTimeDigitized']

project_root = os.path.abspath(os.path.dirname(__file__))
project_root = project_root.replace('\\registration', '')
data_path = os.path.join(project_root, 'data', 'traps', '1', 'IMG_0005.JPG')
exif_data = get_exif_data(data_path)
date2_str = exif_data['DateTimeDigitized']

print(date1_str, date2_str)
minutes_difference = calculate_minutes_difference(date1_str, date2_str)
print(f"Разница в минутах: {minutes_difference}")
