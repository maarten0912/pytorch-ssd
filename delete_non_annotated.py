import argparse
import os
import glob


def rename_files(directory, extension):
    os.chdir(directory)
    for fi in glob.glob("*." + extension.upper()):
        os.rename(fi, fi[:-3] + extension)


def delete_files(path_to_images, path_to_annotations):
    keeping = []
    os.chdir(path_to_annotations)
    for fi in glob.glob("*"):
        if fi.endswith(".xml"):
            keeping.append(os.path.splitext(fi)[0])
        else:
            os.remove(fi)

    os.chdir(path_to_images)
    for fi in glob.glob("*"):
        if not (fi.endswith(".jpg") and os.path.splitext(fi)[0] in keeping):
            os.remove(fi)

    keeping = []
    os.chdir(path_to_images)
    for fi in glob.glob("*"):
        if fi.endswith(".jpg"):
            keeping.append(os.path.splitext(fi)[0])
        else:
            os.remove(fi)

    os.chdir(path_to_annotations)
    for fi in glob.glob("*"):
        if not (fi.endswith(".xml") and os.path.splitext(fi)[0] in keeping):
            os.remove(fi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program that deletes files that are not annotated")
    parser.add_argument("path_to_images", help="Path to folder in which you want to delete non annotated")
    parser.add_argument("path_to_annotations", help="Path to folder in which you want to check if annotated")
    args = parser.parse_args()

    print("RENAMING FILES...")
    rename_files(args.path_to_images, "jpg")
    rename_files(args.path_to_annotations, "xml")

    print("DELETING UNNECESSARY FILES...")
    delete_files(args.path_to_images, args.path_to_annotations)
