- Place a supervise.ly dataset right here.
- a tf_csv folder will be gen here too.
- tensorflow records will be generated from the tf_csv folder.

Like so:

```shell
# first convert supervise.ly's annotations to csv, from root of repo
python input/supervisely2tf_csv.py  # using default values
```

```shell
# Generate `train.record`
python input/generate_tfrecord.py \
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
```