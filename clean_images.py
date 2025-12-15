import os
from PIL import Image

def clean_image_folder(folder):
    valid_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        if not ext in valid_ext or not os.path.isfile(fpath):
            print(f"Removing non-image file: {fpath}")
            os.remove(fpath)
        else:
            try:
                with Image.open(fpath) as img:
                    img.verify()  # Check if image can be opened
            except Exception:
                print(f"Removing corrupted image: {fpath}")
                os.remove(fpath)

clean_image_folder('data/images/authentic')
clean_image_folder('data/images/counterfeit')
print("Cleanup complete.")