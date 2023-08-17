import os

import mlflow

from miscc.config import cfg
from logger import logger


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


def stop_tracking():
    # save logfile
    mlflow.log_artifact(cfg.log_file, "output/log")
    run = mlflow.active_run()
    mlflow.end_run()
    logger.info("Active mlflow run_name:{} run_id: {} started...".format(run.info.run_name, run.info.run_id))


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


def log_model(dir_name, model, model_io_signature):
    if model_io_signature is not None and cfg.framework == "pytorch":
        logger.info("Logging eager-model to mlflow backend...")
        mlflow.pytorch.log_model(
            model,
            "output/%s/model" % dir_name,
            signature=model_io_signature,
            pip_requirements="./requirements.txt",
        )
        logger.info("Done")
        logger.info("Logging script-model to mlflow backend...")
        import torch
        mlflow.pytorch.log_model(
            torch.jit.script(model),
            "output/%s/scripted" % dir_name,
            signature=model_io_signature,
            pip_requirements="./requirements.txt",
        )
        logger.info("Done")
        # torch.onnx.export(self.net, x, "faster_rcnn.onnx", opset_version=11)
