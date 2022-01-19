import argparse
import os
import sys
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from p_tqdm import p_map

def transform(filename):
    if filename.endswith(".txt"):
        path = os.path.join(args.annotations_dir, filename)
        with open(path, 'r') as f:
            root = ET.Element("annotation")

            for line in f:
                data = line.strip().split(' ')

                object = ET.SubElement(root, "object")

                if data[4] != '1':
                    print(f"Expected 1, but was {data[4]}")

                ET.SubElement(object, "name").text = "car"

                bndbox = ET.SubElement(object, "bndbox")
                ET.SubElement(bndbox, "xmin").text = data[0]
                ET.SubElement(bndbox, "ymin").text = data[1]
                ET.SubElement(bndbox, "xmax").text = data[2]
                ET.SubElement(bndbox, "ymax").text = data[3]

            tree = ET.ElementTree(root)
            tree.write(path.split('.')[0] + ".xml")

    else:
        print("[Warning] Found a non-text file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program is to make the CARPK dataset a VOC format")
    parser.add_argument("annotations_dir", metavar="annotations-dir", help="Directory of VOC dataset")
    args = parser.parse_args()

    args.annotations_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.annotations_dir)))

    if not os.path.exists(args.annotations_dir):
        print("Directory does not exist", file=sys.stderr)
        exit(1)

    p_map(transform, os.listdir(args.annotations_dir))

    # with multiprocessing.Pool(10) as p:
    #     r = list(tqdm(p.imap(transform, os.listdir(args.annotations_dir))))

    # for filename in tqdm(os.listdir(args.annotations_dir)):
    #     p = multiprocessing.Process(target=transform, args=(filename,))
    #     p.start


