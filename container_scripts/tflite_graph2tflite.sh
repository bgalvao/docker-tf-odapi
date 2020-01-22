# this is an adaptation of the bazel instructions found here:
# running_on_mobile_tensorflowlite.md
# it has everything you need to know, even how to get a
# frozen graph compatible for tflite conversion

# USER PARAMS -----------------------------------------------------------------
MODEL_NAME="1k"

# DEFAULTS
OUTPUT_DIR=./output/$MODEL_NAME/tflite  # a default you shouldn't need to change

# pick one
# INFERENCE_TYPE=QUANTIZED_UINT8
INFERENCE_TYPE=FLOAT

# If you want to use QUANTIZED_UINT8
# one easier way is to input a *quantized* model

# otherwise, you have to blindly set 
# --default_ranges_min= and --default_ranges_max=
# a sample error message regarding this is below.
# For the most part, INFERENCE_TYPE=FLOAT suffices.

# USER PARAMS END -------------------------------------------------------------


TOCO_PATH="../tensorflow/bazel-bin/tensorflow/lite/toco/toco"

if [ ! -d  $OUTPUT_DIR ]
then
  echo "making output directory $OUTPUT_DIR"
  mkdir -p $OUTPUT_DIR
fi

INPUT_FILE=output/$MODEL_NAME/tflite_graph.pb
OUTPUT_FILE=$OUTPUT_DIR/detect.tflite

INPUT_ARRAYS=normalized_input_image_tensor
OUTPUT_ARRAYS='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',\
'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'


echo -e "\n\nParameter Config:"
echo "- INPUT_FILE :: $INPUT_FILE"
echo "- OUTPUT_FILE :: $OUTPUT_FILE"
echo "- INPUT_ARRAYS :: $INPUT_ARRAYS"
echo "- OUTPUT_ARRAYS :: $OUTPUT_ARRAYS"
echo "- INFERENCE_TYPE :: $INFERENCE_TYPE"


echo -e "converting $INPUT_FILE \n\n"
$TOCO_PATH \
    --input_file=$INPUT_FILE \
    --output_file=$OUTPUT_FILE \
    --input_shapes=1,300,300,3 \
    --input_arrays=$INPUT_ARRAYS \
    --output_arrays=$OUTPUT_ARRAYS \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --inference_type=$INFERENCE_TYPE \
    --allow_custom_ops


# write a labelmap.txt too
cat output/$MODEL_NAME/label_map.pbtxt |\
 grep name | sed s/"\s\sname: '*'"/""/ |\
  sed s/"'"/""/ >> $OUTPUT_DIR/labelmap.txt

chown 1000:1000 -R $OUTPUT_DIR 
chmod 777 $OUTPUT_DIR/*

# -----------------------------------------------------------------------------
# Note that when INFERENCE_TYPE=QUANTIZED_UINT8 without passing
# --default_ranges_min= and --default_ranges_max=
# (found on )
# 
# 2019-09-18 10:52:47.997901: F tensorflow/lite/toco/tooling_util.cc:1728] 
# Array FeatureExtractor/MobilenetV2/Conv/Relu6, which is an input to the 
# DepthwiseConv operator producing the output array 
# FeatureExtractor/MobilenetV2/expanded_conv/depthwise/Relu6, 
# is lacking min/max data, which is necessary for quantization.

# If accuracy matters, either target a non-quantized output format, 
# or run quantized training with your model from a floating point checkpoint 
# to change the input graph to contain min/max information.

# If you don't care about accuracy, you can pass 
# --default_ranges_min= and --default_ranges_max= for easy experimentation.
