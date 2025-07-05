import os
import sys
import argparse
from shutil import rmtree


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig
from global_vars import LOGS_DIR, CHECKPOINTS_DIR

from utils.IO import writeOBJ
import numpy as np
from typing import List

def make_model(config):
    model = NCS(config)
    print("Building model...")
    model.build(input_shape=config.input_shape)
    model.summary()
    print("Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer)
    if config.experiment.checkpoint is not None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, config.experiment.checkpoint)
        model.load_weights(checkpoint_path)
    return model

class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs_to_save: List[int]):
        super().__init__()
        self.epochs_to_save = set(epochs_to_save)
        self.save_dir = './debug_outputs'
        os.makedirs(self.save_dir, exist_ok=True)
        print("Debugging directory:", self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_save:
            save_dir2 = os.path.join(self.save_dir, f"{epoch:05d}")
            os.makedirs(save_dir2, exist_ok=True)
            print("=== Debugging Information ===")
            print(f"Epoch: {epoch}")
            print("Logs:")
            for key, value in logs.items():
                print(f"  - {key}: {value}")
            body = self.model.body
            garment = self.model.garment

            debug_body = self.model.debug_body.numpy()
            debug_vertices = self.model.debug_vertices.numpy()
            debug_unskinned = self.model.debug_unskinned.numpy()
            print(f"  >> body: {debug_body.shape}")
            print(f"  >> garment: {debug_vertices.shape}")
            print(f"  >> unskinned: {debug_unskinned.shape}")
            for batch_idx in range(debug_body.shape[0]):
                save_path = os.path.join(save_dir2, f"{epoch:05d}_{batch_idx:02d}_body.obj")
                writeOBJ(save_path, debug_body[batch_idx].tolist(), body.faces)
                print(f"  [{batch_idx+1}/{debug_body.shape[0]}] saved body to", save_path)

                save_path = os.path.join(save_dir2, f"{epoch:05d}_{batch_idx:02d}_garment.obj")
                garment_vertices = debug_vertices[batch_idx][..., -1].transpose(1,0)
                writeOBJ(save_path, garment_vertices.tolist(), garment.faces)
                print(f"  [{batch_idx+1}/{debug_body.shape[0]}] saved garment to", save_path)

                save_path = os.path.join(save_dir2, f"{epoch:05d}_{batch_idx:02d}_garment_unskinned.obj")
                writeOBJ(save_path, debug_unskinned[batch_idx].tolist(), garment.faces)
                print(f"  [{batch_idx+1}/{debug_body.shape[0]}] saved unskinned garment to", save_path)
            print("=============================")


            
def main(config):
    
    import logging
    tf.get_logger().setLevel("ERROR")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # Remove previous runs logs and checkpoints for this experiment
    log_dir = os.path.join(LOGS_DIR, config.name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, config.name)
    if os.path.isdir(log_dir) or os.path.isdir(checkpoint_dir):
        rmtree(log_dir, ignore_errors=True)
        rmtree(checkpoint_dir, ignore_errors=True)
        print("="*40)
        print(f"WARNING: Removed existing directories for experiment '{config.name}'")
        print("="*40)
        # print(f"There already are logs/checkpoints for an experiment with the same name. Experiment name: {config.name}")
        # print("Please remove or rename the logs/checkpoints. Alternatively, rename the experiment (JSON file name).")
        # return

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing model...")
    if len(gpus) > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = make_model(config)
    else:
        model = make_model(config)

    print("Reading data...")
    data = Data(config, mode="train")
    validation_data = Data(config, mode="validation")

    print("Training...")
    model.fit(
        data,
        validation_data=validation_data,
        epochs=config.experiment.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                write_graph=False,
                write_steps_per_second=False,
                update_freq="epoch",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir, save_freq="epoch"
            ),
            DebugCallback(epochs_to_save=[
                1, 5, 10, 20, 50, 100, 200, 300, 400, 500,
                1000, 1500, 2000, 2500, 3000, 4000, 5000
            ])
        ],
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    opts = parser.parse_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

    # Limit VRAM usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")
        sys.exit()

    config = MainConfig(opts.config)
    main(config)
