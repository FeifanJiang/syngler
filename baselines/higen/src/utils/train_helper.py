import os, yaml
import torch
from utils.arg_helper import edict2dict, dict_to_edict
from easydict import EasyDict as edict
import shutil


def data_to_gpu(*input_data):
    return_data = []
    for dd in input_data:
        if type(dd).__name__ == 'Tensor':
            return_data += [dd.cuda()]

    return tuple(return_data)


def snapshot(model, optimizer, config, step, gpus=[0], tag=None, scheduler=None, scaler=None):
    if scheduler is not None:
        model_snapshot = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step
        }
    else:
        model_snapshot = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step
        }

    if config.use_amp:
        model_snapshot.update({"scaler": scaler.state_dict()})

    # Atomic save: write to temp file then rename (prevents corruption on kill)
    final_path = os.path.join(config.save_dir, "model_snapshot_{}.pth".format(tag)
                 if tag is not None else
                 "model_snapshot_{:07d}.pth".format(step))
    tmp_path = final_path + ".tmp"
    torch.save(model_snapshot, tmp_path)
    os.replace(tmp_path, final_path)

    # Keep only 2 most recent checkpoints to save disk
    import glob
    ckpts = sorted(glob.glob(os.path.join(config.save_dir, "model_snapshot_*.pth")))
    for old_ckpt in ckpts[:-2]:
        try:
            os.remove(old_ckpt)
        except OSError:
            pass

    # to update config file's test path
    # config.config_save_name = os.path.join(config.save_dir, 'config.yaml')
    config_save = dict_to_edict(yaml.safe_load(open(config.config_save_name, 'r')))
    config_save.test.test_model_dir = config.save_dir
    config_save.test.test_model_name = "model_snapshot_{}.pth".format(tag) if tag is not None else \
        "model_snapshot_{:07d}.pth".format(step)

    # update config file's train path
    config_save.train.resume_dir = config_save.test.test_model_dir
    config_save.train.resume_model = config_save.test.test_model_name
    config_save.train.resume_epoch = step
    config_save.train.is_resume = True

    yaml.dump(edict2dict(config_save), open(config.config_save_name, 'w'), default_flow_style=False)
    shutil.copyfile(config.config_save_name, os.path.join(config.save_dir, 'config.yaml'))
    if config.get("sync_fn", None):
        config.sync_fn(config.save_dir)


def load_model(model, file_name, device, optimizer=None, scheduler=None, scaler=None):
    model_snapshot = torch.load(file_name, map_location=device, weights_only=False)
    model.load_state_dict(model_snapshot["model"])
    if optimizer is not None:
        optimizer.load_state_dict(model_snapshot["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(model_snapshot["scheduler"])

    if scaler is not None:
        scaler.load_state_dict(model_snapshot["scaler"])
