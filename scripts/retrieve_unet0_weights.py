# Important: requires TF 2.3.1 (see retrieve_unet0_weights.yml)
import os

from dlc.models.unet0.model import retrieve_model


def main():
    model_dir = os.environ["model_dir"]
    models = (("sahel_sudan_v1_0_0", True), ("sahara_v1_0_0", False))
    for model_name, is_sudan_h5 in models:
        print(f"Processing model: {model_name}")
        model_file = f"{model_dir}/{model_name}/{model_name}.h5"
        weights_file = f"{model_dir}/{model_name}/weights.h5"
        model = retrieve_model(model_file, is_sudan_h5=is_sudan_h5)
        model.summary()
        print("Saving weights...")
        model.save_weights(weights_file)

    print("Finished")


if __name__ == "__main__":
    main()
