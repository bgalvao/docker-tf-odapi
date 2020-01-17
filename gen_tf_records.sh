# first convert supervise.ly's annotations to csv
python input/supervisely2tf_csv.py \
--supervisely_dir=./input/supervisely \
--output_dir=./input/tf_csv

# Generate `train.record`
python ./utils/data/generate_tfrecord.py \
--csv_input=input/tf_csv/train_labels.csv \
--output_path=input/tf_records/train.record \
--img_path=input/tf_csv/images/train \
--label_map input/tf_csv/label_map.pbtxt


# Generate `test.record`
python ./utils/data/generate_tfrecord.py \
--csv_input=input/tf_csv/test_labels.csv \
--output_path=input/tf_records/test.record \
--img_path=input/tf_csv/images/test \
--label_map input/tf_csv/label_map.pbtxt
