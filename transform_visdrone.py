import argparse
import os
import sys
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from p_tqdm import p_map, p_umap, p_imap, p_uimap


categories = {
    '0': 'ignored region',
    '1': 'pedestrian',
    '2': 'person',
    '3': 'bicycle',
    '4': 'car',
    '5': 'van',
    '6': 'truck',
    '7': 'tricycle',
    '8': 'awning tricycle',
    '9': 'bus',
    '10': 'motor',
    '11': 'other'
}

def transform(filename):
    if filename.endswith(".txt"):
        path = os.path.join(args.annotations_dir, filename)
        with open(path, 'r') as f:
            root = ET.Element("annotation")

            for line in f:
                data = line.strip().split(',')

                if data[4] == '0':
                    #print("[Warning] Got bounding box with score of 0, ignoring")
                    continue

                try:
                    category = categories[data[5]]
                except KeyError:
                    print(f"[Warning] Found invalid category: {data[5]}")
                    continue

                object = ET.SubElement(root, "object")

                ET.SubElement(object, "name").text = category

                box_xmin = int(data[0])
                box_ymin = int(data[1])
                box_width = int(data[2])
                box_height = int(data[3])

                bndbox = ET.SubElement(object, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(box_xmin)
                ET.SubElement(bndbox, "ymin").text = str(box_ymin)
                ET.SubElement(bndbox, "xmax").text = str(box_xmin + box_width)
                ET.SubElement(bndbox, "ymax").text = str(box_ymin + box_height)

                ET.SubElement(object, "truncation").text = data[6]
                ET.SubElement(object, "occlusion").text = data[7]

            tree = ET.ElementTree(root)
            tree.write(path.split('.')[0] + ".xml")

    else:
        print("[Warning] Found a non-text file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program is to make the VisDrone dataset a VOC format")
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


