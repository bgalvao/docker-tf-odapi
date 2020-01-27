from os import listdir
from os.path import join, exists

from bullet import Bullet, colors, styles

from export import export


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


# pick model
available_models = [
    model for model in listdir('./output') if '.git' not in model
]
model = Bullet(
    choices=available_models,
    prompt="\nChoose from one of the models:",
    **ocean_style_whale
).launch()


# pick checkpoint
training_dir = join('output', model, 'training')
checkpoints = [
    ckpt.replace('.meta', '') 
    for ckpt in listdir(training_dir) if '.meta' in ckpt and \
        ckpt.replace('.meta', '') != 'model.ckpt'
]
ckpt_prefix = Bullet(
    prompt="Choose checkpoint to export the model from:",
    choices=['[ last checkpoint ]'] + checkpoints,
    **ocean_style_dolphin
).launch()
# the function pipeline.set_training_config below figures this out on its own.
ckpt_prefix = None if ckpt_prefix == '[ last checkpoint ]' else ckpt_prefix


export(model, ckpt_prefix=ckpt_prefix)