import gzip
import shutil

def extract_gzip_file(compressed_file, output_file):
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    compress_file = "C:\\Users\\1000250081\\_work\\data\\Candela\\samples\\Test_PSc.img"
    out_file = "gzip.txt"



    extract_gzip_file(compress_file, out_file)