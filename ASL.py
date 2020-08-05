import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug
from imgaug import augmenters as iaa
ROOT_DIR = os.path.abspath('./')
print(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
############################################################
#  Configurations
############################################################


class GestureConfig(Config):
    """Configuration for training on the ASL  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ASL"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # Background + gesture

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    #Learning rate changes
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.001


############################################################
#  Dataset
############################################################

class GestureDataset(utils.Dataset):

    def load_gesture(self, dataset_dir, subset):
        """Load a subset of the ASL dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only 6 classes to add.
        self.add_class("ASL", 1, "all2")
        self.add_class("ASL", 2,"accident1")
        self.add_class("ASL", 3, "adopt")
        self.add_class("ASL", 4, "advise")
        self.add_class("ASL", 5, "again")
        self.add_class("ASL", 6, "airplane")
        self.add_class("ASL", 7, "call_on_phone")
        self.add_class("ASL", 8, "chicago")
        self.add_class("ASL", 9, "all_right")
        self.add_class("ASL", 10, "any")
        self.add_class("ASL", 11, "answer-fr")
        self.add_class("ASL", 12, "apple")
        self.add_class("ASL", 13, "appointment")
        self.add_class("ASL", 14, "art")
        self.add_class("ASL", 15, "approximately")
        self.add_class("ASL", 16, "article")
        self.add_class("ASL", 17, "awful2")
        self.add_class("ASL", 18, "awkward")
        self.add_class("ASL", 19, "atlanta")
        self.add_class("ASL", 20, "baseball")
        self.add_class("ASL", 21, "become")
        self.add_class("ASL", 22, "bad")
        self.add_class("ASL", 23, "behavior-n1")
        self.add_class("ASL", 24, "baltimore")
        self.add_class("ASL", 25, "bread")
        self.add_class("ASL", 26, "beer")
        self.add_class("ASL", 27, "breakfast")
        self.add_class("ASL", 28, "bird")
        self.add_class("ASL", 29, "busy2")
        self.add_class("ASL", 30, "blue")
        self.add_class("ASL", 31, "buy")
        self.add_class("ASL", 32, "bored")
        self.add_class("ASL", 33, "camping")
        self.add_class("ASL", 34, "can1")
        self.add_class("ASL", 35, "boss2")
        self.add_class("ASL", 36, "boston")
        self.add_class("ASL", 37, "center1")
        self.add_class("ASL", 38, "brown")
        self.add_class("ASL", 39, "chat")
        self.add_class("ASL", 40, "california")
        
        # Train or validation dataset?
        assert subset in ["train", "val","test1"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '81.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annot = open(os.path.join(dataset_dir, "via_project_region.json"))
        annotations = json.load(annot)
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            named = [y['region_attributes'] for y in a['regions']]
            class_ids = [int(n['ASL']) for n in named]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "ASL",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids = class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a ASL dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ASL":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids,dtype = np.int32)
        return mask, class_ids#np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ASL":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GestureDataset()
    dataset_train.load_gesture(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GestureDataset()
    dataset_val.load_gesture(args.dataset, "val")
    dataset_val.prepare()
    seq = iaa.Sequential([iaa.ChangeColorTemperature((1100,10000)),
    iaa.Fliplr(0.9),iaa.Flipud(0.8)])
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=800,
                layers='all') 
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect gestures.')
    parser.add_argument("command",
                        metavar="<command>",
                        help='train')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/gesture/dataset/",
                        help='Directory of the gesture dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GestureConfig()
    else:
        class InferenceConfig(GestureConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
