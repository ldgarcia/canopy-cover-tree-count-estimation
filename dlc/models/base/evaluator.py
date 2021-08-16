import abc
import importlib
import os
import time
from typing import Optional

import geopandas as gpd
import tensorflow as tf
import yaml

import dlc.random
import dlc.tools.config
import dlc.tools.datasets
import dlc.tools.images
from dlc.models.base.settings import Settings


__all__ = ["Evaluator"]


class Evaluator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        settings: Settings,
        tpu_strategy=None,
        gpu_number: Optional[int] = None,
    ):
        self.settings = settings
        self.model: Optional[tf.keras.Model] = None
        self._tpu_strategy = tpu_strategy
        self._image_loader = None
        self._image_ds_gen = None
        self._image_cache = None
        self.gpu_number = gpu_number
        if self.gpu_number is None:
            self.gpu_number = os.environ.get("gpu_number", None)
            if self.gpu_number is not None:
                self.gpu_number = int(self.gpu_number)

    def _load_model(self) -> tf.keras.Model:
        print(f"importing module...: dlc.models.{self.settings.model}")
        module = importlib.import_module(f"dlc.models.{self.settings.model}")
        load_model = getattr(module, "load_model")
        model = load_model(
            input_shape=self.settings.input_shape,
            batch_size=self.settings.training_batch_size,
            weights_file=self.settings.get_weights_file(),
            **dict(self.settings),
        )
        model.compile(
            optimizer=self.settings.get_optimizer(),
            loss=self.settings.get_loss(),
            loss_weights=self.settings.loss_weights,
            metrics=self.settings.get_metrics(),
            run_eagerly=self.settings.run_eagerly,
        )
        return model

    def _load_frames(self) -> gpd.GeoDataFrame:
        frames = gpd.read_file(self.settings.frames_path)
        frames_query = self.settings.evaluation_frames_query
        if frames_query is not None:
            print(f"Selecting from {len(frames)} frame rows...")
            print(f"Executing query: {frames_query}")
            frames.query(frames_query, inplace=True)
        print(f"Selected {len(frames)} frame rows.")
        return frames

    def _pre_run(self):
        frames = self._load_frames()
        self._image_ds_gen = dlc.tools.datasets.ImageDatasetGenerator(
            frames,
            image_keys=(
                self.settings.feature_keys,
                self.settings.annotation_keys,
            ),
            input_base_path=self.settings.frames_directory,
        )
        self._image_cache = dlc.tools.cache.ArrayCache()
        p = self.settings.testing_local_standardization_p
        self._image_loader = dlc.tools.images.ImageLoader(
            local_standardization_p=p,
            cache=self._image_cache,
            seed=self.settings.seed,
            defaults=self.settings.loader_patch_defaults,
        )
        print("Preloading cache with test sample frames...")
        self._image_loader.preload_cache(
            self._image_ds_gen.get_cache_preload_generator(),
        )
        cache_size = self._image_cache.size * 1e-6
        print(f"Finished preloading cache [{cache_size:.2f} Mb total size]")

    def _post_run(self):
        self._image_cache.clear()

    def _create_test_ds(self):
        ds = self._image_ds_gen.get_sequential_patches(
            self.settings.patch_size,
            patch_stride=self.settings.patch_stride,
            shuffle=False,
            seed=self.settings.seed,
            verbose=self.settings.verbose,
        )
        ds = ds.map(
            self._image_loader.load,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        for transformer in self.settings.get_transformers():
            ds = ds.map(
                transformer,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        ds = ds.batch(self.settings.testing_batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _evaluate(self, ds) -> object:
        results = self._model.evaluate(
            ds,
            steps=self.settings.testing_steps,
            verbose=1,
            return_dict=True,
        )
        return results

    def run(self):
        # Optionally restrict visibility of GPUs
        if self.gpu_number is not None:
            # From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    tf.config.set_visible_devices(gpus[self.gpu_number], "GPU")
                    print("Physical GPUS:\n", gpus)
                    logical_gpus = tf.config.list_logical_devices("GPU")
                    print("Logical GPUS:\n", logical_gpus)
                except RuntimeError as e:
                    print(
                        "Visible devices must be set before GPUs have been initialized"
                    )
                    print(e)

        # Set global seeds
        tf.keras.backend.clear_session()
        dlc.random.check_environment()
        dlc.random.set_seeds(self.settings.seed)

        # Check that if we are running in GPU environment,
        # as signaled by the CUDA_VISIBLE_DEVICES variable,
        # then TF can see the device.
        gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if gpu_dev is not None:
            devices = tf.config.list_physical_devices("GPU")
            if len(devices) > 0:
                print(f"TF reports as visible GPUs {devices}.")
            else:
                raise Exception("TF can't access a GPU device.")

        # Execute pre-run logic (e.g. preload cache)
        self._pre_run()

        try:
            print("About to load model...")
            start_time = time.perf_counter()
            # Instantiate model
            self._model = self._load_model()
            print("Model loaded")
            # Create output directory
            self.settings.out_dir.mkdir(parents=True, exist_ok=True)
            # Evaluate the model
            print("About to evaluate model...")
            ds = self._create_test_ds()
            results = self._evaluate(ds)
            # Record the metrics/losses
            path = self.settings.out_dir.joinpath(f"test_eval.yml")
            print(f"Will write results to: {path}")
            with open(path, "w") as file:
                file.write(yaml.dump(results))
            elapsed_time = (time.perf_counter() - start_time) / 60
            print(f"Run took {elapsed_time} minutes")
        finally:
            # Execute post-run logic (e.g. clear cache)
            self._post_run()
            tf.keras.backend.clear_session()


def main():
    config = dlc.tools.config.read_config()
    setting_dict = config["settings"]
    module = importlib.import_module(f"dlc.models.{setting_dict['model']}")
    settings_t = getattr(module, "Settings", Settings)
    evaluator_t = getattr(module, "Evaluator", Evaluator)
    settings = settings_t(**setting_dict)
    evaluator = evaluator_t(settings)
    evaluator.run()


if __name__ == "__main__":
    main()
