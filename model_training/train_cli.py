from os import listdir
from os.path import exists, join

from bullet import VerticalPrompt, Bullet, styles, colors, Input, YesNo, Numbers

from tf_org_models import downloadable_models
from load import load_model_paths
from pipeline import set_training_config
from train import train




white = colors.foreground['white']
alignment = {'indent':1, 'shift':1, 'align':2, 'margin':2, 'pad_right':2}

whale_bullet = " 🐋"
ocean_style_whale = styles.Ocean.copy()
ocean_style_whale.update({
    'bullet':whale_bullet, 'bullet_color':white,
    'word_on_switch':white
})
ocean_style_whale.update(alignment)

dolphin_bullet = " 🐬"
ocean_style_dolphin = styles.Ocean.copy()
ocean_style_dolphin.update({
    'bullet':dolphin_bullet, 'bullet_color':white,
    'word_on_switch':white
})
ocean_style_dolphin.update(alignment)



available_models = [
    model for model in listdir('./output') if '.git' not in model
]
download_choice = '[ Download a new model ]'


model = Bullet(
    choices=available_models + [download_choice],
    prompt="\nChoose from one of the models:",
    **ocean_style_whale
).launch()

model2resume = None
model2download = None
alt_model_name = None

if model is download_choice:
    cli = Bullet(
        prompt="Model to download:",
        choices=list(downloadable_models.keys()),
        **ocean_style_dolphin
    )
    model2download = cli.launch()
    
    alt_model_name_cli = Input(
        prompt="\n There's already a \'{}\' in your disk.\
        \n Please, type a different name: \n\n ".format(model2download), 
        strip=True, indent=alignment['indent']
    )

    if model2download in available_models:
        alt_model_name = alt_model_name_cli.launch()
    else:
        cli = YesNo(
            prompt="\n The model to download is going to be named \'{}\'. \
                \n Do you want to change this name? [y/n] ".format(
                    model2download
                ),
            prompt_prefix="",
            indent=alignment['indent']
        )
        if cli.launch():
            alt_model_name_cli = Input(
                prompt="\n Type a new name: \n ",
                strip=True, indent=alignment['indent']
            )
            alt_model_name = alt_model_name_cli.launch()

else:
    model2resume = model


print(' Loading model...')
training_dir, _, base_pipeline_config_path = load_model_paths(
    model2resume, model2download, alt_model_name
)

checkpoints = [
    ckpt.replace('.meta', '') 
    for ckpt in listdir(training_dir) if '.meta' in ckpt
]

ckpt_prefix = Bullet(
    prompt="Choose checkpoint to resume from:",
    choices=['[ last checkpoint ]'] + checkpoints,
    **ocean_style_dolphin
).launch()
# the function pipeline.set_training_config below figures this out on its own.
ckpt_prefix = None if ckpt_prefix == '[ last checkpoint ]' else ckpt_prefix

num_steps = Numbers(
    prompt="\nPick a number of training epochs: ",
    type=int,
    indent=alignment['indent']
).launch()

num_eval_steps = Numbers(
    prompt="\nEnter max number of evaluations: ",
    type=int,
    indent=alignment['indent']
).launch()

batch_size = Numbers(
    prompt="\nEnter desired batch size: ",
    type=int,
    indent=alignment['indent']
).launch()

training_config_file_path = join(training_dir, 'training.config')
print('\n⇒ writing training config to {}'.format(training_config_file_path))

# defaults :: current input dataset converted to tfrecords
test_record_fname = './input/tf_records/test.record'
train_record_fname = './input/tf_records/train.record'
label_map_pbtxt_fname = './input/tf_csv/label_map.pbtxt'

training_config_file = set_training_config(
    base_pipeline_config=base_pipeline_config_path,
    model_name='ssd_mobilenet_v2',
    train_record=train_record_fname, 
    test_record=test_record_fname,
    labelmap_pbtxt_path=label_map_pbtxt_fname,  # later copied to export dir
    batch_size=batch_size,
    num_steps=num_steps,
    num_eval_steps=num_eval_steps,
    checkpoint=ckpt_prefix
)

train_intent = YesNo(
    prompt="Do you want to train in this session? [y/n]: ",
    prompt_prefix=""
).launch()

if train_intent:
    print('\n🐬 🐬 🐬 "training will now commence!" 🐬 🐬 🐬')
    print('\n🐋 :: "I\'ll see you on the other side fren" :: 🐋\n')
                
    train(training_config_file, training_dir, num_steps, num_eval_steps)