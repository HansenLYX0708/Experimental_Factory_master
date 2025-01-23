from PIL import Image
import tifffile

file_path = "C:\\Users\\1000250081\\_work\\data\AFM_Raw_file\\4Normal\\062_R001_C001_01_Forward.tiff"

try:
    # Using Pillow for basic information
    with Image.open(file_path) as img:
        print("Format:", img.format)
        print("Mode:", img.mode)
        print("Size:", img.size)
        print("Info:", img.info)

    # Using tifffile for advanced analysis
    with tifffile.TiffFile(file_path) as tiff:
        print("Number of Pages:", len(tiff.pages))
        print("Tags of First Page:")
        for tag in tiff.pages[0].tags.values():
            print(f"{tag.name}: {tag.value}")
except Exception as e:
    print("Error:", e)
