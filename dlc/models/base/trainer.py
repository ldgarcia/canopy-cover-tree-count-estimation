import abc
import datetime
import gc
import hashlib
import importlib
import json
import os
import pathlib
import shutil
import time
from typing import List
from typing import Optional

import geopandas as gpd
import git
import pandas as pd
import tensorflow as tf
import yaml

import dlc.callbacks
import dlc.random
import dlc.tools.config
import dlc.tools.datasets
import dlc.tools.images
import dlc.tools.splits
from dlc.models.base.settings import Settings

# This model has a custom Model wrapper

__all__ = ["Trainer", "run_trainer"]


class Trainer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        settings: Settings,
        tpu_strategy=None,
        is_sanity_check: Optional[bool] = None,
        is_retraining: Optional[bool] = None,
        use_tensorboard: Optional[bool] = None,
        epochs: Optional[int] = None,
        gpu_number: Optional[int] = None,
    ):
        self.settings = settings
        self._model: Optional[tf.keras.Model] = None
        self._tpu_strategy = tpu_strategy
        self._training_image_loader = None
        self._validation_image_loader = None
        self._image_ds_gen = None
        self._image_cache = None
        self._resume_from_epoch = None

        # The following properties are read from the environment
        self.is_sanity_check = is_sanity_check
        if self.is_sanity_check is None:
            self.is_sanity_check = bool(os.environ.get("is_sanity_check", False))
        self.is_retraining = is_retraining
        if self.is_retraining is None:
            self.is_retraining = bool(os.environ.get("is_retraining", False))
        self.epochs = epochs
        if self.epochs is None:
            self.epochs = int(os.environ.get("epochs", 1))
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard is None:
            self.use_tensorboard = bool(os.environ.get("use_tensorboard", False))
        self.gpu_number = gpu_number
        if self.gpu_number is None:
            self.gpu_number = os.environ.get("gpu_number", None)
            if self.gpu_number is not None:
                self.gpu_number = int(self.gpu_number)

    def _create_model(self) -> tf.keras.Model:
        model = None
        module = importlib.import_module(f"dlc.models.{self.settings.model}")
        loss = self.settings.get_loss()
        metrics = self.settings.get_metrics()
        if not self.is_retraining:
            create_model = getattr(module, "create_model")
            model = create_model(
                input_shape=self.settings.input_shape,
                batch_size=self.settings.training_batch_size,
                **dict(self.settings),
            )
            model.compile(
                optimizer=self.settings.get_optimizer(),
                loss=loss,
                loss_weights=self.settings.loss_weights,
                metrics=metrics,
                run_eagerly=self.settings.run_eagerly,
            )
        else:
            model_path = self.settings.out_dir.joinpath("model.h5")
            # Using workaround suggested in:
            # https://github.com/tensorflow/tensorflow/issues/45903
            custom_objects = self._get_custom_objects(
                loss=loss,
                metrics=metrics,
                optimizer=self.settings.get_optimizer(),
                # Objects in the custom_objects dictionary only get used
                # if required, so no harm is done by adding here those
                # that will be needed by some of the models.
                other={
                    # Loss weights need to be passed as custom object.
                    "loss_weights": self.settings.loss_weights,
                },
            )
            model_path = self.settings.out_dir.joinpath("model.h5")
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=True,
            )
            # Using workaround suggested in:
            # https://github.com/tensorflow/tensorflow/issues/45903
            model.compile(
                loss=model.loss,
                loss_weights=self.settings.loss_weights,
                optimizer=model.optimizer,
                metrics=metrics,
            )
        return model

    def _get_custom_objects(
        self,
        *,
        loss,
        metrics,
        optimizer,
        other: Optional[dict] = None,
    ) -> dict:
        custom_objects = {}
        xs: list = list(loss.values()) if isinstance(loss, dict) else [loss]
        if isinstance(metrics, dict):
            for ys in metrics.values():
                xs.extend(ys)
        else:
            xs.extend(metrics)
        for x in xs:
            if x is not None:
                if isinstance(x, tf.keras.losses.Loss) or isinstance(
                    x, tf.keras.metrics.Metric
                ):
                    custom_objects[type(x).__name__] = x
                else:
                    custom_objects[x.__name__] = x
        custom_objects[type(optimizer).__name__] = optimizer
        if other is not None:
            return {**custom_objects, **other}
        return custom_objects

    def _get_last_epoch(self) -> int:
        csv_path = self.settings.out_dir.joinpath("training.csv")
        df = pd.read_csv(csv_path, delimiter=",")
        if len(df) > 0:
            return len(df) - 1
        return 1

    def _get_best_monitor_score(self) -> float:
        csv_path = self.settings.out_dir.joinpath("training.csv")
        results = pd.read_csv(csv_path, delimiter=",")
        if self.settings.checkpoint_mode == "min":
            return results[self.settings.checkpoint_monitor].min()
        return results[self.settings.checkpoint_monitor].max()

    def _backup_model(self, n) -> None:
        fnames = [
            "model.h5",
            "random_state.pickle",
        ]
        for fname in fnames:
            src = self.settings.out_dir.joinpath(fname)
            dst = self.settings.out_dir.joinpath(f"{n}-{fname}")
            dst_s256 = self.settings.out_dir.joinpath(f"{n}-{fname}.shasum256")
            shutil.copy(src, dst)
            computed_hash = None
            with open(src, "rb") as file:
                bytes = file.read()
                computed_hash = hashlib.sha256(bytes).hexdigest()
            with open(dst_s256, "w") as out_file:
                txt = f"{computed_hash}\n"
                out_file.write(txt)
            print(f"Backed-up file: {fname}")

    def _check_manifest(self) -> None:
        manifest_path = self.settings.out_dir.joinpath("manifest.yml")
        if not manifest_path.exists():
            print("Warning: no manifest file.")
            return

        fnames_sha256 = {}
        with open(manifest_path, "r") as manifest_file:
            manifest = yaml.load(manifest_file)
            fnames_sha256 = manifest["files"]
        for fname, hash in fnames_sha256.items():
            computed_hash = None
            with open(self.settings.out_dir.joinpath(fname), "rb") as file:
                bytes = file.read()
                computed_hash = hashlib.sha256(bytes).hexdigest()
            if hash != computed_hash:
                raise Exception(f"Invalid sha256 for file: {fname}")
        print("Completed manifest check.")

    def _write_manifest(self) -> None:
        fnames = [
            "settings.yml",
            "params.yml",
            "model.json",
            "model.h5",
            "weights.h5",
            "training.csv",
            "random_state.pickle",
        ]
        fnames_sha256 = {}
        for fname in fnames:
            path = self.settings.out_dir.joinpath(fname)
            if not path.exists():
                print("Skipping missing file: {fname}")
                continue
            checksum = None
            with open(path, "rb") as file:
                bytes = file.read()
                checksum = hashlib.sha256(bytes).hexdigest()
            fnames_sha256[fname] = checksum

        git_revision = None
        repo_path = None
        try:
            tmp = os.getenv("DLC_PROJECT_DIRECTORY")
            if tmp is not None:
                repo_path = pathlib.Path(tmp)
                if repo_path.exists():
                    repo = git.Repo(path=repo_path)
                    git_revision = repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            print(f"Invalid Git repo: {repo_path}")

        manifest = {
            "git_revision": git_revision,
            "files": fnames_sha256,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        manifest_path = self.settings.out_dir.joinpath("manifest.yml")
        with open(manifest_path, "w") as manifest_file:
            manifest_file.write(yaml.dump(manifest))

    def _save_model(self):
        path = self.settings.out_dir.joinpath("model.h5")
        self._model.save(path, save_format="h5")

    def _load_frames(self) -> gpd.GeoDataFrame:
        frames = gpd.read_file(self.settings.frames_path)
        frames_query = self.settings.frames_query
        if frames_query is not None:
            print(f"Selecting from {len(frames)} frame rows...")
            print(f"Executing query: {frames_query}")
            frames.query(frames_query, inplace=True)
        print(f"Selected {len(frames)} frame rows.")
        return frames

    def _pre_run(self):
        frames = self._load_frames()
        splitter = self.settings.get_splitter()
        if not hasattr(splitter, self.settings.splitter.method):
            raise ValueError("Invalid splitter method")
        split_method = getattr(splitter, self.settings.splitter.method)
        splits, _ = split_method(
            frames,
            seed=self.settings.splitter_seed,
            **self.settings.splitter.method_parameters,
        )
        print(f"CV split index: {self.settings.cross_validation_split_index}")
        if self.settings.cross_validation_split_index is not None:
            k = self.settings.cross_validation_split_index
            print(f"Selecting CV split: {k}")
            splits = splits[k]
            print(f"Split has {len(splits)} elements.")

        self._image_ds_gen = dlc.tools.datasets.ImageDatasetGenerator(
            frames,
            splits=splits,
            splits_map=splitter.get_splits_map(),
            image_keys=(
                self.settings.feature_keys,
                self.settings.annotation_keys,
            ),
            input_base_path=self.settings.frames_directory,
            seed=self.settings.seed,
        )
        # Optionally, load canopy cover prior for output bias initialization
        if self.settings.use_target_class_prior:
            print(f"Will use target class prior")
            if self.settings.target_class_prior is None:
                frames["split"] = splits
                frames.query(
                    "split == {}".format(
                        splitter.get_splits_map()["training"],
                    )
                )
                prior = frames["canopy_cover"].median()
                print(f"Using computed prior with value: {prior:.6f}")
                self.settings.target_class_prior = prior
                frames.drop(columns=["split"], inplace=True)
            else:
                prior = self.settings.target_class_prior
                print(f"Using given prior with value: {prior:.6f}")
        # Note: We can use a single cache for both loaders
        #       since the splits use independent frames.
        self._image_cache = dlc.tools.cache.ArrayCache()
        p = self.settings.training_local_standardization_p
        self._training_image_loader = dlc.tools.images.ImageLoader(
            local_standardization_p=p,
            cache=self._image_cache,
            seed=self.settings.seed,
            defaults=self.settings.loader_patch_defaults,
        )
        p = self.settings.validation_local_standardization_p
        self._validation_image_loader = dlc.tools.images.ImageLoader(
            local_standardization_p=p,
            cache=self._image_cache,
            seed=self.settings.seed,
            defaults=self.settings.loader_patch_defaults,
        )
        print("Preloading cache with training sample frames...")
        self._training_image_loader.preload_cache(
            self._image_ds_gen.get_cache_preload_generator(split="training"),
        )
        print("Preloading cache with validation sample frames...")
        self._validation_image_loader.preload_cache(
            self._image_ds_gen.get_cache_preload_generator(split="test")
        )
        cache_size = self._image_cache.size * 1e-6
        print(f"Finished preloading cache [{cache_size:.2f} Mb total size]")

    def _post_run(self):
        self._image_cache.clear()

    def _create_train_val_ddss(self):
        training_ds = None
        if not self.is_sanity_check:
            print("Initializing training random patches generator...")
            training_ds = self._image_ds_gen.get_random_patches(
                self.settings.patch_size,
                split="training",
                seed=self.settings.seed,
                verbose=self.settings.verbose,
            )
        else:
            print(
                "Sanity check.",
                "Initializing training sequential generator...",
            )
            training_ds = self._image_ds_gen.get_sequential_patches(
                self.settings.patch_size,
                split="training",
                shuffle=False,
                seed=self.settings.seed,
                verbose=self.settings.verbose,
                return_cardinality=False,
            )
            # See:
            # https://cs231n.github.io/neural-networks-3/#before-learning-sanity-checks-tipstricks
            training_ds = training_ds.take(20)
            training_ds = training_ds.repeat()

        training_ds = training_ds.map(
            self._training_image_loader.load,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        augmenter = self.settings.get_augmenter()
        if augmenter is not None:
            training_ds = training_ds.map(
                augmenter,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        for transformer in self.settings.get_transformers():
            training_ds = training_ds.map(
                transformer,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if not self.is_sanity_check:
            training_ds = training_ds.batch(self.settings.training_batch_size)
        else:
            training_ds = training_ds.batch(1)

        training_ds = training_ds.prefetch(tf.data.AUTOTUNE)

        print("Initializing validation sequential patches generator...")
        cardinality, validation_ds = self._image_ds_gen.get_sequential_patches(
            self.settings.patch_size,
            split="test",
            shuffle=False,
            seed=self.settings.seed,
            verbose=self.settings.verbose,
            return_cardinality=True,
        )
        validation_steps = cardinality / self.settings.validation_batch_size
        if self.settings.validation_steps is not None:
            print("Checking that specified # of validation steps is correct")
            if validation_steps < self.settings.validation_steps:
                raise ValueError("Invalid # of validation steps")
        else:
            print(f"Using computed # of validation steps: {validation_steps}")
            self.settings.validation_steps = validation_steps

        if self.settings.validation_n_skip is not None:
            validation_ds = validation_ds.skip(self.settings.validation_n_skip)

        validation_ds = validation_ds.map(
            self._validation_image_loader.load,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        for transformer in self.settings.get_transformers():
            validation_ds = validation_ds.map(
                transformer,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        validation_ds = validation_ds.batch(
            self.settings.validation_batch_size,
        )
        validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

        return training_ds, validation_ds

    def _train(self, training_ds, validation_ds, initial_epoch, epochs):
        callbacks = self._create_callbacks(validation_data=validation_ds)
        steps_per_epoch = self.settings.training_steps
        if self.is_sanity_check:
            steps_per_epoch = 20

        if not self.is_retraining and self.settings.use_lsuv_initialization:
            from dlc.initializers import lsuv_initialization

            lsuv_initialization(
                self._model,
                training_ds.take(1),
                tol_var=self.settings.lsuv_tol_var,
                max_iter=self.settings.lsuv_max_iter,
                verbose=True,
            )

        history = None
        if not self.is_sanity_check:
            history = self._model.fit(
                training_ds,
                initial_epoch=initial_epoch,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_ds,
                validation_steps=self.settings.validation_steps,
                callbacks=callbacks,
                verbose=1,
            )
            for cb in callbacks:
                if isinstance(cb, dlc.callbacks.BaseBatchPredictionPlotCallback):
                    mean, std = cb.compute_time_statistics()
                    print(f"Timing information for callback {type(cb).__name__}:")
                    print(f"mean: {mean} s, std: {std}")
        else:
            history = self._model.fit(
                training_ds,
                initial_epoch=initial_epoch,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1,
            )
        return history

    def _eval(self, val2_ds) -> tf.keras.Model:
        # Load model with best weights
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
        # Evaluate it on val2 dataset
        results = model.evaluate(
            val2_ds,
            steps=self.settings.validation_steps,
            verbose=1,
            return_dict=True,
        )
        return results

    def _create_callbacks(
        self,
        *,
        validation_data=None,
    ) -> List[tf.keras.callbacks.Callback]:
        # set prior to being called.
        callbacks = []

        # Optional tensorboard
        if self.use_tensorboard:
            tensorboard_cb = tf.keras.callbacks.TensorBoard(
                log_dir="./logs",
                histogram_freq=1,
            )
            callbacks.append(tensorboard_cb)

        # 1) Model checkpointing
        if not (self.is_sanity_check or self.settings.is_lr_range_test):
            if os.environ.get("run_profiler", False):
                log_dir = self.settings.out_dir.joinpath("profiling")
                os.makedirs(log_dir, exist_ok=True)
                profiling_cb = tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    write_steps_per_second=True,
                    update_freq="epoch",
                    profile_batch="2, 10",
                )
                callbacks.append(profiling_cb)

            previous_best = None
            if self.is_retraining:
                previous_best = self._get_best_monitor_score()

            checkpoint_path = str(self.settings.out_dir.joinpath("model.h5"))
            bw_checkpoint_path = str(self.settings.out_dir.joinpath("weights.h5"))

            print(f"Will save model checkpoints to {checkpoint_path}")

            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=self.settings.checkpoint_monitor,
                mode=self.settings.checkpoint_mode,
                verbose=1,
                save_best_only=False,
                save_weights_only=False,
            )
            callbacks.append(checkpoint_cb)

            bw_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                bw_checkpoint_path,
                monitor=self.settings.checkpoint_monitor,
                mode=self.settings.checkpoint_mode,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            )
            if previous_best is not None:
                bw_checkpoint_cb.best = previous_best
            callbacks.append(bw_checkpoint_cb)

            if validation_data is not None:
                validation_output_path = self.settings.out_dir.joinpath(
                    "plots/validation"
                )
                print(
                    "Will save validation plots to {}".format(
                        validation_output_path,
                    )
                )
                kwargs = dict(
                    cmap=self.settings.validation_plot_cmap,
                    output_path=validation_output_path,
                    monitor=self.settings.checkpoint_monitor,
                    mode=self.settings.checkpoint_mode,
                    save_best_only=self.settings.validation_plot_save_best_only,
                    n_max_samples=self.settings.validation_plot_n_max_samples,
                    n_take=self.settings.validation_plot_n_take,
                    n_skip=self.settings.validation_plot_n_skip,
                    show_hist=self.settings.validation_plot_show_hist,
                )
                if self.settings.validation_plot_keys is None:
                    val_plot_cb = dlc.callbacks.BatchPredictionPlotCallback(
                        validation_data,
                        **kwargs,
                    )
                    if previous_best is not None:
                        val_plot_cb.best = previous_best
                    callbacks.append(val_plot_cb)
                else:
                    val_plot_cb = dlc.callbacks.MultiOutputBatchPredictionPlotCallback(
                        validation_data,
                        keys=self.settings.validation_plot_keys,
                        **kwargs,
                    )
                    if previous_best is not None:
                        val_plot_cb.best = previous_best
                    callbacks.append(val_plot_cb)
        # 2) Garbage collection
        gc_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda _batch, _logs: gc.collect()
        )
        callbacks.append(gc_callback)
        # 3) Training log
        logger_callback = tf.keras.callbacks.CSVLogger(
            str(self.settings.csv_training_log_file),
            append=self.is_retraining,
        )
        callbacks.append(logger_callback)
        return callbacks

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
        if not self.is_retraining:
            dlc.random.set_seeds(self.settings.seed)
        else:
            print("Will retrain. Loading cached random state.")
            state_file = self.settings.out_dir.joinpath("random_state.pickle")
            seeds = dlc.random.load_state(state_file)
            self.settings.seed, self.settings.splitter_seed = seeds

        # Execute pre-run logic (e.g. preload cache)
        self._pre_run()

        try:
            start_time = time.perf_counter()
            # Instantiate model
            if self._tpu_strategy is None:
                self._model = self._create_model()
            else:
                with self._tpu_strategy.scope():
                    self._model = self._create_model()

            # Create output directory
            if not self.is_retraining:
                manifest_path = self.settings.out_dir.joinpath("manifest.json")
                exists_ok = not manifest_path.exists()
                self.settings.out_dir.mkdir(
                    parents=True,
                    exist_ok=exists_ok,
                )

            # If resuming training, check manifest and get last epoch
            initial_epoch = 0
            epochs = self.epochs
            if self.is_retraining:
                self._check_manifest()
                initial_epoch = self._get_last_epoch() + 1
                self._backup_model(initial_epoch)
                epochs = initial_epoch + self.epochs

            # Train the model
            training_ds, validation_ds = self._create_train_val_ddss()

            # Record the model settings
            print("Saving settings.")
            path = self.settings.out_dir.joinpath(f"settings.yml")
            with open(path, "w") as file:
                file.write(yaml.dump(self.settings.dict()))

            print("Saving model architecture.")
            path = self.settings.out_dir.joinpath(f"model.json")
            with open(path, "w") as file:
                json.dump(self._model.to_json(), file)

            print(f"Will now begin training loop, epochs: [{initial_epoch}, {epochs})")
            history = self._train(
                training_ds,
                validation_ds,
                initial_epoch,
                epochs,
            )
            print(f"Training finished.")

            # Save other model information
            print("Saving params.")
            path = self.settings.out_dir.joinpath(f"params.yml")
            with open(path, "w") as file:
                file.write(yaml.dump(history.params))

            print("Saving full model.")
            self._save_model()

            # Save random state
            print("Saving random state.")
            state_file = self.settings.out_dir.joinpath("random_state.pickle")
            seeds = (self.settings.seed, self.settings.splitter_seed)
            dlc.random.save_state(seeds, state_file)

            # Record hashes in manifest file
            print("Writing manifest file.")
            self._write_manifest()
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
    trainer_t = getattr(module, "Trainer", Trainer)
    settings = settings_t(**setting_dict)
    trainer = trainer_t(settings)
    trainer.run()


if __name__ == "__main__":
    main()
