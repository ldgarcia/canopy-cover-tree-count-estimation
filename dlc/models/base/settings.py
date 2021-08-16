import importlib
import pathlib
from inspect import isclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pydantic
import tensorflow as tf
import tensorflow_addons as tfa

from dlc.tools.images import DefaultsSpec
from dlc.tools.plots import ColorMapSpec
from dlc.tools.splits import ObjectSplitter

__all__ = ["Settings"]


LossType = Union[tf.keras.losses.Loss, Callable]
MetricType = Union[tf.keras.metrics.Metric, Callable]


class OptimizerSpec(pydantic.BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]]

    def new_instance(self) -> tf.keras.optimizers.Optimizer:
        module = importlib.import_module("dlc.optimizers")
        opt_t = getattr(module, self.name)
        opt = None
        if self.parameters is not None:
            if isinstance(self.parameters["learning_rate"], dict):
                print("Using custom LearningRateSchedule")
                lr_spec = self.parameters["learning_rate"]
                custom_sched_mod = importlib.import_module("dlc.lr_schedules")
                if hasattr(custom_sched_mod, lr_spec["name"]):  # local
                    lr_t = getattr(
                        custom_sched_mod,
                        lr_spec["name"],
                    )
                elif hasattr(tfa.optimizers, lr_spec["name"]):  # TFA's
                    lr_t = getattr(
                        tfa.optimizers,
                        lr_spec["name"],
                    )
                else:  # TF's
                    lr_t = getattr(
                        tf.keras.optimizers.schedules,
                        lr_spec["name"],
                    )
                lr = lr_t(**lr_spec["parameters"])
                self.parameters["learning_rate"] = lr
            opt = opt_t(**self.parameters)
        else:
            opt = opt_t()
        assert isinstance(opt, tf.keras.optimizers.Optimizer)
        return opt


class LossSpec(pydantic.BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]]

    def new_instance(self) -> LossType:
        module = importlib.import_module("dlc.losses")
        loss_t = getattr(module, self.name)
        if isclass(loss_t) and issubclass(loss_t, tf.keras.losses.Loss):
            if self.parameters is not None:
                return loss_t(**self.parameters)
            else:
                return loss_t()
        return loss_t


class MetricSpec(pydantic.BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]]

    def new_instance(self) -> MetricType:
        module = importlib.import_module("dlc.metrics")
        metric_t = getattr(module, self.name)
        if isclass(metric_t) and issubclass(metric_t, tf.keras.metrics.Metric):
            if self.parameters is not None:
                return metric_t(**self.parameters)
            else:
                return metric_t()
        return metric_t


class TransformerSpec(pydantic.BaseModel):
    name: str

    def new_instance(self) -> Callable:
        module = importlib.import_module("dlc.transformers")
        transformer = getattr(module, self.name)
        assert callable(transformer)
        return transformer


class SplitterSpec(pydantic.BaseModel):
    name: str
    parameters: Dict[str, Any] = {}
    method: str
    method_parameters: Dict[str, Any]

    def new_instance(self) -> ObjectSplitter:
        module = importlib.import_module("dlc.tools.splits")
        splitter_t = getattr(module, self.name)
        instance = splitter_t(**self.parameters)
        return instance


class Settings(pydantic.BaseSettings):
    """Defines the settings common to all the trainers \
        (also compatible with evaluators)."""

    class Config:
        case_sensitive = True
        extra = "ignore"

        # Give highest priority to init_settings.
        @classmethod
        def customise_sources(
            cls,
            init_settings: pydantic.env_settings.SettingsSourceCallable,
            env_settings: pydantic.env_settings.SettingsSourceCallable,
            file_secret_settings: pydantic.env_settings.SettingsSourceCallable,
        ) -> Tuple[pydantic.env_settings.SettingsSourceCallable, ...]:
            return init_settings, env_settings, file_secret_settings

    model: str
    optimizer: OptimizerSpec
    is_lr_range_test: bool = False
    loss: Union[LossSpec, Dict[str, Optional[LossSpec]]]
    loss_weights: Optional[Dict[str, float]]
    metrics: Union[List[MetricSpec], Dict[str, List[MetricSpec]]]
    transformers: Optional[List[TransformerSpec]]
    kernel_regularizer: Optional[str] = None  # for weight decay
    model_name_suffix: Optional[str] = None
    dataset_name: str
    data_directory: pydantic.DirectoryPath
    input_dtype: str = "float32"
    input_channels: int
    output_channels: int
    output_name: str = "output"
    seed: Optional[int] = None
    use_orthogonal_initialization: bool = False
    use_lsuv_initialization: bool = False
    lsuv_tol_var: float = 1e-5
    lsuv_max_iter: int = 25
    use_target_class_prior: bool = False
    target_class_prior: Optional[float] = None
    target_class_prior_scale: float = 1.0
    frames_file: pathlib.Path
    frames_query: Optional[str] = None
    evaluation_frames_query: Optional[str] = None
    feature_keys: List[str] = ["image"]
    augmenter: Optional[str] = None
    annotation_keys: List[str]
    loader_patch_defaults: DefaultsSpec = None
    splitter_seed: Optional[int] = None
    verbose: bool = True
    patch_size: Tuple[int, int]
    patch_stride: Optional[Tuple[int, int]] = None
    training_local_standardization_p: Tuple[Optional[float], Optional[float]] = (
        0.4,
        None,
    )
    validation_local_standardization_p: Tuple[Optional[float], Optional[float]] = (
        0.0,
        None,
    )
    testing_local_standardization_p: Tuple[Optional[float], Optional[float]] = (
        0.0,
        None,
    )

    run_eagerly: bool = True
    training_batch_size: int
    training_steps: int
    validation_batch_size: int
    validation_steps: Optional[int] = None
    validation_n_skip: Optional[int] = None
    testing_batch_size: int
    testing_steps: Optional[int] = None
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    validation_plot_cmap: Tuple[ColorMapSpec, ...] = ()
    validation_plot_save_best_only: bool = True
    validation_plot_n_max_samples: int = 8
    validation_plot_n_take: int
    validation_plot_n_skip: int = 0
    validation_plot_keys: Optional[Tuple[str, ...]] = None
    validation_plot_show_hist: bool = True

    best_plot_n_take: int = 5
    best_plot_n_skip: int = 0

    # By default, use a train-validation split.
    splitter: SplitterSpec = SplitterSpec(
        name="LatitudeObjectSplitter",
        method="train_val_split",
        method_parameters={"val_size": 0.2},
    )
    # Indicates which cross valdation split to use,
    # when the spliting method is 'stratified_k_fold'.
    cross_validation_split_index: Optional[int] = None

    # @pydantic.validator("cross_validation_split_index")
    # def validate_cross_validation_split_index(
    #    cls,
    #    cross_validation_split_index: int,
    # ):
    #    if cross_validation_split_index is not None:
    #        if cls.splitter.method not in [
    #            "stratified_k_fold",
    #            "train_val_split_k_fold",
    #        ]:
    #            raise ValueError("Using CV but splitter method is invalid'")
    #        if "n_splits" not in cls.splitter.method_parameters:
    #            raise ValueError("Using CV but number of splits is unspecified")
    #        if (
    #            cross_validation_split_index
    #            >= cls.splitter.method_parameters["n_splits"]
    #        ):
    #            raise ValueError("Invalid CV index for the specified number of splits")
    #    return cross_validation_split_index

    plots: Tuple[Dict[str, Any], ...] = ()

    base_weights: Optional[str] = None
    custom_objects: Optional[Dict] = None

    def get_weights_file(self, filename: str = "weights.h5") -> pathlib.Path:
        if self.base_weights is not None:
            return self.models_directory.joinpath(
                self.base_weights,
                filename,
            )
        return self.out_dir.joinpath(filename)

    def get_optimizer(self):
        return self.optimizer.new_instance()

    def get_splitter(self):
        return self.splitter.new_instance()

    def get_loss(self):
        if isinstance(self.loss, dict):
            return {
                k: (v.new_instance() if v is not None else v)
                for k, v in self.loss.items()
            }
        return self.loss.new_instance()

    def get_metrics(self):
        if isinstance(self.metrics, list):
            return [m.new_instance() for m in self.metrics]
        return {k: [m.new_instance() for m in v] for k, v in self.metrics.items()}

    def get_augmenter(self):
        if self.augmenter is not None:
            module = importlib.import_module("dlc.models.augmentation")
            augmenter_t = getattr(module, self.augmenter)
            return augmenter_t()
        else:
            return None

    def get_transformers(self):
        if self.transformers is not None:
            return [t.new_instance() for t in self.transformers]
        else:
            return []

    @property
    def input_shape(self):
        return (*self.patch_size, self.input_channels)

    @property
    def run_id(self) -> str:
        base = [self.model]
        if self.model_name_suffix is not None:
            base.append(self.model_name_suffix)
        base.append(self.dataset_name)
        base.append(f"{self.seed}")
        return "-".join(base)

    alt_models_directory: Optional[str] = None

    @property
    def models_directory(self) -> pathlib.Path:
        if self.alt_models_directory is None:
            return self.data_directory.joinpath("models")
        return self.data_directory.joinpath(self.alt_models_directory)

    @property
    def out_dir(self) -> pathlib.Path:
        tmp = str(self.data_directory)
        dir = pathlib.Path(tmp)
        return dir.joinpath(self.models_directory, self.run_id)

    @property
    def csv_training_log_file(self) -> pathlib.Path:
        return self.out_dir.joinpath("training.csv")

    @property
    def frames_path(self) -> pathlib.Path:
        return self.data_directory.joinpath(
            "datasets/frames",
            self.frames_file,
        )

    @property
    def frames_directory(self) -> pathlib.Path:
        return self.frames_path.parent

    def __init__(self, **kwargs):
        super(Settings, self).__init__(**kwargs)
