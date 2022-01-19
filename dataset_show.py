import os
import glob
import random
import cv2
import argparse
from xml.etree import cElementTree as ElementTree
import numpy as np
import colorsys

def randColor():
	h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
	return [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]

parser = argparse.ArgumentParser(description="Displays annotations over images")
parser.add_argument("dataset_dir", metavar="dataset-dir", help="Directory of VOC dataset")
parser.add_argument('--verified', type=int, default=-1, help="1 -> show only verified images. 0 -> show only unverified images")
args = parser.parse_args()


colors = {}

scaling = None

args.dataset_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.dataset_dir)))
annotations = glob.glob(os.path.join(args.dataset_dir, 'Annotations', '*.xml'))
print(f"{len(annotations)} annotations found")

i = 0
while True:
	annotationPath = annotations[i]
	tree = ElementTree.parse(annotationPath)
	root = tree.getroot()

	if args.verified != -1 and args.verified != int(root.find('is_verified_by_human').text):
		continue

	
	imagePathList = glob.glob(annotationPath.replace('Annotations','JPEGImages').replace('.xml','.*'))
	if len(imagePathList) == 0:
		print(f'Found no images for \'{annotationPath}\': could not find \'{os.path.join(annotationPath.replace("Annotations","JPEGImages"), ".*")}\'')
		continue
	elif len(imagePathList) > 1:
		print(f'Found multiple images for \'{annotationPath}\', taking first one: \'{imagePathList[0]}\'')
	image = cv2.imread(imagePathList[0])

	if image is None:
		print(f"Annotation {annotationPath[60:]} has no image.")
		continue

	height, width, _ = image.shape
	if scaling is None: scaling = max(1.5*width/1920, 1.5*height/1080)
	image = cv2.resize(image, (int(width / scaling), int(height / scaling)))
	h,w, _ = image.shape
	bigger = np.zeros((h+len(colors)*30,w,3), dtype=np.uint8)
	bigger[len(colors)*30:h+len(colors)*30,:] = image
	image = bigger


	for obj in root.findall('object'):
		className = obj.find('name').text
		if className not in colors:
			colors[className] = randColor()
			h,w, _ = image.shape
			bigger = np.zeros((h+30,w,3), dtype=np.uint8)
			bigger[30:h+30,:] = image
			image = bigger

		color = colors[className]

		box = obj.find('bndbox')
		fields = ["xmin", "ymin", "xmax", "ymax"]
		xmin, ymin, xmax, ymax = [int(float(box.find(field).text) / scaling) for field in fields]
		o = len(colors)*30
		cv2.rectangle(image, (xmin, ymin+o, xmax-xmin, ymax-ymin), color, 3)

	
	for x in range(len(colors)):
		cv2.putText(image, list(colors.items())[x][0], (10, x*30), cv2.FONT_HERSHEY_PLAIN, 2.0, list(colors.items())[x][1], 2)


	cv2.imshow("image", image)
	cv2.setWindowTitle("image", annotationPath)
	print(annotationPath)
	k = cv2.waitKey(0)
	if k == 81 or k == 8 or k == 101:
		i -= 1
	elif k == 27:
		exit()
	else:
		i = (i + 1) % len(annotations)
