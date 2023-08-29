import datetime
import os

import PIL.Image
import dateutil.tz
import mlflow
import numpy as np
import torch
from torchvision.transforms.transforms import _setup_size

from miscc.config import cfg
from logger import logger

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
log_file = "./log_%s.txt" % timestamp


def start_tracking():
    keys = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]
    for key in keys:
        assert os.environ[key] != "", "'%s' is required to set in environment" % keys
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name="%s_%s" % (cfg.CONFIG_NAME, cfg.DATASET_NAME))
    # mlflow.autolog(log_models=True)
    # dagshub.init(os.environ['DAGSHUB_REPO'], os.environ['DAGSHUB_USERNAME'], mlflow=True)
    mlflow.start_run(description=cfg.experiment_description)
    run = mlflow.active_run()
    # track additional hparmas
    track_config()
    logger.info("Active mlflow run_name:{} run_id: {} started...".format(run.info.run_name, run.info.run_id))


def stop_tracking(exit_message="ended peacefully"):
    global log_file
    # save logfile
    mlflow.log_artifact(log_file, "output/log")
    mlflow.log_param("exit_message", exit_message)
    run = mlflow.active_run()
    mlflow.end_run()
    logger.info("Active mlflow run_name:{} run_id: {} stopped".format(run.info.run_name, run.info.run_id))


def prepare_for_mlflow(**kwargs):
    big_dict = {}
    for key, dicts in kwargs.items():
        for k, v in dicts.items():
            big_dict["%s.%s" % (key, k)] = v
    return big_dict


def track_config():
    d = dict(cfg)
    del d["TREE"]
    del d["TRAIN"]
    del d["TEXT"]
    if "GAN" in cfg.keys():
        del d["GAN"]
    mlflow.log_params(d)
    if "GAN" in cfg.keys():
        mlflow.log_params(
            prepare_for_mlflow(
                TREE=cfg.TREE, TRAIN=cfg.TRAIN, TEXT=cfg.TEXT, GAN=cfg.GAN
            )
        )
    else:
        mlflow.log_params(
            prepare_for_mlflow(
                TREE=cfg.TREE, TRAIN=cfg.TRAIN, TEXT=cfg.TEXT
            )
        )


def save_image(epoch_idx, train_losses, train_performance_dict):
    from utilities import save_images, tensor_to_pil
    
    if self.sample_output is not None and self.sample_input is not None:
        filename = "epoch_output_%d.png" % epoch_idx
        local_path = self.cfg.output_dir / filename
        save_images(
            tensor_to_pil(self.sample_input),
            tensor_to_pil(self.sample_output),
            local_path,
            "PNG",
            "epoch_idx=%d, Loss [%0.4e], %s:%0.4e"
            % (
                epoch_idx,
                self.accumulate_loss(train_losses),
                self.cfg.train.metrics,
                train_performance_dict[self.cfg.train.metrics],
            ),
        )
        # save image as mlflow artifact
        logger.info("Saving Artifact:%s" % local_path)
        mlflow.log_artifact(str(local_path), "output/images")
        os.remove(local_path)
        self.sample_input = None
        self.sample_output = None


def log_model(op_root, dir_name, model, model_io_signature):
    logger.info("Saving local model: '%s'" % ("%s/%s" % (op_root, dir_name)))
    torch.save(model.state_dict(), "%s/%s" % (op_root, dir_name))
    try:
        logger.info("Logging eager-model to mlflow backend...'%s'" % ("output/%s/model" % dir_name))
        mlflow.pytorch.log_model(
            model,
            "output/model/%s" % dir_name,
        )
        logger.info("Done")
        logger.info('Save G/Ds models.')
    except Exception as e:
        logger.exception(e)
        ...


def except_hook(cls, exception, traceback):
    """Give us back the original exception hook that may have changed"""
    from logger import logger
    
    logger.exception(cls)
    logger.exception(exception)
    mlflow.log_param("Exception", "%s, %s, %s" % (cls, exception, traceback))
    stop_tracking("ended with exception")


class AspectResize(torch.nn.Module):
    """
   Resize image while keeping the aspect ratio.
   Extra parts will be covered with 255(white) color value
   """
    
    def __init__(self, size, background=255):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.background = background
    
    @staticmethod
    def fit_image_to_canvas(image: PIL.Image, canvas_width, canvas_height, background=255) -> PIL.Image:
        # Get the dimensions of the image
        image_width, image_height = image.size
        
        # Calculate the aspect ratio of the image
        image_aspect_ratio = image_width / float(image_height)
        
        # Calculate the aspect ratio of the canvas
        canvas_aspect_ratio = canvas_width / float(canvas_height)
        
        # Calculate the new dimensions of the image to fit the canvas
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = canvas_height * image_aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / image_aspect_ratio
        
        # Resize the image to the new dimensions
        image = image.resize((int(new_width), int(new_height)), PIL.Image.BICUBIC)
        
        # Create a blank canvas of the specified size
        canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
        canvas[:, :, :] = background
        
        # Calculate the position to paste the resized image on the canvas
        x = int((canvas_width - new_width) / 2)
        y = int((canvas_height - new_height) / 2)
        
        # Paste the resized image onto the canvas
        canvas[y:y + int(new_height), x:x + int(new_width)] = np.array(image)
        
        return PIL.Image.fromarray(canvas)
    
    def forward(self, image: PIL.Image) -> PIL.Image:
        image = self.fit_image_to_canvas(image, self.size[0], self.size[1], self.background)
        return image
