from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
import argparse
import os
import glob

def sanitizePixel(x, y, shape):
    max_x = shape[1]
    max_y = shape[0]
    return (int(min(max_x,max(0,x))),int(min(max_y,max(0,y))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Displays annotations over images")
    parser.add_argument("net_type", metavar="net-type", help="The network type, one of vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite and sq-ssd-lite")
    parser.add_argument("model_path", metavar="model-path", help="Path of model")
    parser.add_argument("label_path", metavar="label-path", help="Path of label file")
    parser.add_argument("images_path", metavar="images-path", help="Directory with images")
    args = parser.parse_args()

    if len(sys.argv) < 5:
        print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
        sys.exit(0)
    net_type = args.net_type
    model_path = args.model_path
    label_path = args.label_path
    image_path = args.images_path

    class_names = [name.strip() for name in open(label_path).readlines()]

    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'vgg16-ssd-512':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite-512':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'sq-dataset_dir':
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, vgg16-ssd-512, mb2-ssd-lite-512 and sq-ssd-lite.")
        sys.exit(1)
    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'vgg16-ssd-512':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=torch.device('cuda:0'))
    elif net_type == 'mb2-ssd-lite-512':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=torch.device('cuda:0'))
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)


    args.images_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.images_path)))
    imageList = glob.glob(os.path.join(args.images_path, "*"))
    print(len(imageList))
    j = 0
    while True:
        orig_image = cv2.imread(imageList[j])
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # top_k = 10
        # prop_threshold = 0.4
        boxes, labels, probs = predictor.predict(image, 10, 0.4)

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            print(f'box={box}')
            cv2.rectangle(orig_image, sanitizePixel(box[0], box[1], image.shape), sanitizePixel(box[2], box[3], image.shape), (255, 255, 0), 2)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        sanitizePixel(box[0], box[1] - 10, image.shape),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # font scale
                        (255, 0, 255),
                        2)  # line type
        
        cv2.imshow("image", orig_image)
        cv2.setWindowTitle("image",imageList[j])
        k = cv2.waitKey(0)
        if k == 27:
            exit()
        elif k == 81 or k == 8 or k == 101:
            j = (j - 1) % len(imageList)
        else:
            j = (j + 1) % len(imageList)
