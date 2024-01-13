from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import tensorrt as trt
import argparse
import json
import numpy as np
import pandas as pd
from baskerville import seqnn


precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

# For TF-TRT:


class OptimizedModel:
    """
    Class of model optimized with tensorrt
    Args:
      saved_model_dir: Folder with saved model
    """

    def __init__(self, saved_model_dir=None):
        self.loaded_model_fn = None

        if not saved_model_dir is None:
            self.load_model(saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model_fn is None:
            raise (Exception("Haven't loaded a model"))
        x = tf.constant(input_data.astype("float32"))
        labeling = self.loaded_model_fn(x)
        try:
            preds = labeling["predictions"].numpy()
        except:
            try:
                preds = labeling["probs"].numpy()
            except:
                try:
                    preds = labeling[next(iter(labeling.keys()))]
                except:
                    raise (
                        Exception("Failed to get predictions from saved model object")
                    )
        return tf.squeeze(preds, axis=0)

    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(
            saved_model_dir, tags=[tag_constants.SERVING]
        )
        wrapper_fp32 = saved_model_loaded.signatures["serving_default"]
        self.loaded_model_fn = wrapper_fp32

    def __call__(self, input_data):
        return self.loaded_model_fn.predict(input_data)


class ModelOptimizer:
    """
    Class of converter for tensorrt
    Args:
      input_saved_model_dir: Folder with saved model of the input model
    """

    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        self.loaded_model = None

        if not calibration_data is None:
            self.set_calibration_data(calibration_data)

    def set_calibration_data(self, calibration_data):
        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype("float32")),)

        self.calibration_data = calibration_input_fn

    def convert(
        self,
        output_saved_model_dir,
        precision="FP32",
        max_workspace_size_bytes=8000000000,
        **kwargs,
    ):
        if precision == "INT8" and self.calibration_data is None:
            raise (Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt_precision,
            max_workspace_size_bytes=max_workspace_size_bytes,
            use_calibration=precision == "INT8",
        )
        converter = tf_trt.TrtGraphConverterV2(
            input_saved_model_dir=self.input_saved_model_dir,
            conversion_params=conversion_params,
        )

        if precision == "INT8":
            converter.convert(calibration_input_fn=self.calibration_data)
        else:
            converter.convert()

        converter.save(output_saved_model_dir=output_saved_model_dir)

        return OptimizedModel(output_saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model is None:
            self.load_default_model()

        return self.loaded_model.predict(input_data)

    def load_default_model(self):
        self.loaded_model = tf.keras.models.load_model("resnet50_saved_model")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a seqnn model to TensorRT model."
    )
    parser.add_argument("model_fn", type=str, help="Path to the Keras model file (.h5)")
    parser.add_argument("params_fn", type=str, help="Path to the JSON parameters file")
    parser.add_argument(
        "targets_file", type=str, help="Path to the target variants file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for storing saved models (original & converted)",
    )
    args = parser.parse_args()

    # Load target variants
    targets_df = pd.read_csv(args.targets_file, sep="\t", index_col=0)

    # Load parameters
    with open(args.params_fn) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    # Load keras model into seqnn class
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_fn)
    seqnn_model.build_slice(np.array(targets_df.index))
    seqnn_model.build_ensemble(True)

    # save this model to a directory
    seqnn_model.model.save(f"{args.output_dir}/original_model")

    # Convert the model
    opt_model = ModelOptimizer(f"{args.output_dir}/original_model")
    model_fp32 = opt_model.convert(f"{args.output_dir}/model_FP32", precision="FP32")


if __name__ == "__main__":
    main()
