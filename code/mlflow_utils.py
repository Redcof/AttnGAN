import os

import PIL.Image
import mlflow
import numpy as np
import torch
from easydict import EasyDict
from mlflow.utils.file_utils import read_yaml
from torchvision.transforms.transforms import _setup_size

from miscc.config import cfg
from logger import logger

model_log_config = EasyDict(dict(model_saved_earlier=False, bast_value=None))


def start_tracking():
    model_log_config.model_saved_earlier = False
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


def is_early_stop(epoch_idx=-1):
    """Early stop"""
    cfg_earl_stp = EasyDict(read_yaml(".", "earlystop.yaml"))
    if cfg_earl_stp.hard_stop_now:
        logger.info("Early stopping...HARD STOP.")
        mlflow.log_params(dict(early_stop="hard stop"))
        return True
    if cfg_earl_stp.stop_at_epoch <= epoch_idx and cfg_earl_stp.stop_at_epoch != -1:
        logger.info("Early stopping for EPOCH STOP.")
        mlflow.log_params(
            dict(early_stop="requested epoch=%d, actual=%d" % (cfg_earl_stp.stop_at_epoch, epoch_idx))
        )
        return True
    if cfg_earl_stp.stop_at_epoch > cfg.train.max_epoch:
        logger.info("EPOCH EXTENDED form:%d to:%d" % (cfg.TRAIN.MAX_EPOCH, cfg_earl_stp.stop_at_epoch))
        cfg.TRAIN.MAX_EPOCH = cfg_earl_stp.stop_at_epoch
    return False


def can_i_log_model(epoch_idx, loss=None, train_performance=None, test_performance=None):
    schedule_type = cfg.train.save_schedule.type
    schedule_key = cfg.train.save_schedule.key
    schedule_delta = cfg.train.save_schedule.value_delta
    threshold = cfg.train.save_schedule.threshold
    flag = False
    if schedule_type == "epoch":
        if epoch_idx % schedule_key == 0:
            model_log_config.model_saved_earlier = True
            flag = True
        else:
            flag = False
    elif schedule_type == "metric":
        # get the current value
        current_value = (
            train_performance[schedule_key]
            if (train_performance and schedule_key in train_performance)
            else (
                test_performance[schedule_key] if (test_performance and schedule_key in test_performance) else None
            )
        )
        if current_value is not None:
            if model_log_config.bast_value is None:
                # Last value none indicated the training just started.
                # set the last_value to current_value
                model_log_config.bast_value = current_value
            if (
                current_value >= threshold
                and current_value > model_log_config.bast_value
                and abs(current_value - model_log_config.bast_value) >= schedule_delta
            ):
                model_log_config.model_saved_earlier = True
                model_log_config.bast_value = current_value
                flag = True
    elif schedule_type == "loss":
        current_value = loss.item() if loss is not None else None
        if current_value is None:
            flag = False
        else:
            if model_log_config.bast_value is None:
                # Last value none indicated the training just started.
                # set the last_value to current_value
                model_log_config.bast_value = current_value
            if (
                current_value <= threshold
                and current_value < model_log_config.bast_value
                and abs(current_value - model_log_config.bast_value) >= schedule_delta
            ):
                model_log_config.model_saved_earlier = True
                model_log_config.bast_value = current_value
                flag = True
    if flag is False and (loss is None and train_performance is None and test_performance is None):
        # log every 5 epoch
        if epoch_idx % 5 == 0:
            flag = True
        else:
            flag = False
    return flag


def stop_tracking(exit_message="ended peacefully"):
    try:
        mlflow.log_artifact(cfg.log_file, "output/log")
        mlflow.log_param("log_file", cfg.log_file)
    except FileNotFoundError as e:
        mlflow.log_param("log_file", str(e))
    mlflow.log_param("exit_message", exit_message)
    run = mlflow.active_run()
    mlflow.end_run()
    logger.info("Active mlflow run_name:{} run_id: {} stopped".format(run.info.run_name, run.info.run_id))
    model_log_config.model_saved_earlier = False


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


def log_model(op_root, name, model, model_io_signature):
    logger.info("Saving local model: '%s'" % ("%s/%s" % (op_root, name)))
    torch.save(model.state_dict(), "%s/%s" % (op_root, name))
    try:
        logger.info("Logging eager-model to mlflow backend...'%s'" % ("output/%s/model" % name))
        mlflow.pytorch.log_model(
            model,
            "output/model/%s" % name,
        )
        logger.info("Done")
        logger.info('Save G/Ds "%s" models.' % name)
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
