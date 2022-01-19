import argparse
import os
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program will train an ONNX model using the mb2-ssd-lite Neural Net")
    parser.add_argument("dataset_dir", metavar="dataset-dir", help="Directory of VOC dataset")
    parser.add_argument("model_dir", metavar="model-dir", help='Directory for saving checkpoint models')
    parser.add_argument("pretrained", help="Pre-trained base model path")
    parser.add_argument("--model-name","-m", metavar="N", help="Name for output model file")
    parser.add_argument('--num-epochs', '-e', default=30, type=int, help='The number epochs')
    parser.add_argument('--batch-size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--finalize', '-f', action='store_true', help='Only convert model to ONNX and put in rar')
    
    args = parser.parse_args()

    exit = False
    args.dataset_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.dataset_dir)))
    args.model_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.model_dir)))


    if not os.path.exists(args.dataset_dir):
        print("Please supply a valid directory", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(args.dataset_dir, "Annotations")):
        print("Missing Annotations directory", file=sys.stderr)
        exit = True
    if not os.path.exists(os.path.join(args.dataset_dir, "ImageSets")):
        print("Missing ImageSets directory", file=sys.stderr)
        exit = True
    if not os.path.exists(os.path.join(args.dataset_dir, "JPEGImages")):
        print("Missing JPEGImages directory", file=sys.stderr)
        exit = True
    if not os.path.exists(os.path.join(args.dataset_dir, "ImageSets", "Main")):
        print(f"Missing {os.path.join(args.dataset_dir, 'ImageSets', 'Main')} directory", file=sys.stderr)
        exit = True
    if not os.path.exists(os.path.join(args.dataset_dir, "labels.txt")):
        print("Missing labels.txt file", file=sys.stderr)
        exit = True
    if exit == True:
        sys.exit(1)

    if args.model_name:
        output = os.path.join(args.model_dir, f'{args.model_name}.onnx')
        rarfolder_name = args.model_name
    else:
        output = os.path.join(args.model_dir,'model.onnx')
        rarfolder_name = 'model'

    try:
        if not args.finalize:
            if args.resume:
                subprocess.run([f"python3 '{os.path.abspath(os.path.join(__file__,os.pardir,'train_ssd.py'))}' --dataset-type=voc --data='{args.dataset_dir}' --model-dir='{args.model_dir}' --resume={args.resume} --net=mb2-ssd-lite --batch-size {args.batch_size} --num-epochs {args.num_epochs}"], shell=True, check=True)
            else:
                subprocess.run([f"python3 '{os.path.abspath(os.path.join(__file__,os.pardir,'train_ssd.py'))}' --dataset-type=voc --data='{args.dataset_dir}' --model-dir='{args.model_dir}' --net=mb2-ssd-lite --pretrained-ssd={args.pretrained} --batch-size {args.batch_size} --num-epochs {args.num_epochs}"], shell=True, check=True)

        subprocess.run([f"python3 '{os.path.abspath(os.path.join(__file__,os.pardir,'onnx_export.py'))}' --model-dir='{args.model_dir}' --net=mb2-ssd-lite --output='{output}'"], shell=True, check=True)

        subprocess.run([f"cd '{args.model_dir}'; mkdir '{os.path.join(args.model_dir,rarfolder_name)}'; cp '{output}' 'labels.txt' '{os.path.join(args.model_dir,rarfolder_name)}'; rar a '{os.path.join(args.model_dir,rarfolder_name)}.rar' '{rarfolder_name}'; rm -r '{os.path.join(args.model_dir,rarfolder_name)}'"], shell=True)
    except subprocess.CalledProcessError:
        sys.exit(1)
