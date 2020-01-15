# Generate `train.record`
!python generate_tfrecord.py \
--csv_input=dataset/train_labels.csv \
--output_path=dataset/train.record \
--img_path=dataset/images/train \
--label_map dataset/label_map.pbtxt


# Generate `test.record`
!python generate_tfrecord.py \
--csv_input=dataset/test_labels.csv \
--output_path=dataset/test.record \
--img_path=dataset/images/test \
--label_map dataset/label_map.pbtxt
