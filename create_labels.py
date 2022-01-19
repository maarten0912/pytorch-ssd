import xml.etree.ElementTree as ET
import os
import argparse


def get_labels(path_to_annotations):
    labels = []

    for filename in os.listdir(path_to_annotations):
        if filename.endswith(".xml"):
            file = os.path.join(path_to_annotations, filename)
            tree = ET.parse(file)
            root = tree.getroot()
            for name in root.iter("name"):
                if name.text not in labels:
                    labels.append(name.text)

    return labels


def create_labels_txt(path_to_annotations):
    labels = get_labels(path_to_annotations)
    txt = open("labels.txt", "w+")
    for i in range(len(labels)):
        if (len(labels) > 1 and i == 0) or (i < len(labels) - 1):
            txt.write(labels[i] + "\n")
        else:
            txt.write(labels[i])
    txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program that creates a labels.txt file")
    parser.add_argument("path", help="Path to the annotations folder containing xml files")
    args = parser.parse_args()
    create_labels_txt(args.path)
