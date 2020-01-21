# convert a saved_model to tensorflow js

# USER PARAMS
MODEL_NAME="sample_model"

# DEFAULTS
MODEL_DIR=./output/$MODEL_NAME
SAVED_MODEL_DIR=$MODEL_DIR/saved_model
OUTPUT_DIR=$MODEL_DIR/tensorflow_js

# safety checks
if [ ! -d $MODEL_DIR ]
then
  echo -e "The model and its associated directory \n\n\
  $MODEL_DIR\n\n do not exist."
fi 

if [ ! -f $MODEL_DIR/saved_model.pb ]
then
  if [ ! -f $SAVED_MODEL_DIR/saved_model.pb ]
    then
      echo -e "saved_model.pb not available\n\n \
      you can try an alternative method using tensorflowjs_wizard"
      exit -3
  fi
fi

echo -e "preparing to convert saved_model.pb...\n\n"
mkdir -p $SAVED_MODEL_DIR/variables
cp $MODEL_DIR/saved_model.pb $SAVED_MODEL_DIR/saved_model.pb

echo -e "running tensorflowjs_converter with proposed default params of \
\`tensorflowjs_wizard\` (try it!)...\n\n"

tensorflowjs_converter \
--input_format=tf_saved_model \
--saved_model_tags=serve \
--signature_name=serving_default \
--strip_debug_ops=True \
$SAVED_MODEL_DIR \
$OUTPUT_DIR

rm -rf $SAVED_MODEL_DIR
