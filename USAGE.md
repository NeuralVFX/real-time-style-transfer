
## Getting Started
- Install `pytorch 0.4.0` and dependencies from https://pytorch.org/
- Install packages `tqdm`, `cv2`, `matplotlib`, `tar`

- Clone this repo:
```bash
git clone https://github.com/NeuralVFX/real-time-style-transfer.git
cd real-time-style-transfer
```
- Download the dataset (e.g. [Pascal VOC 2007](http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar)):
```bash
bash data/get_test_dataset.sh
```
- Download a style image (e.g. [Francis Picabia - "Tarentelle"](https://en.wikipedia.org/wiki/Abstract_art#/media/File:Francis_Picabia,_1912,_Tarentelle,_oil_on_canvas,_73.6_x_92.1_cm,_Museum_of_Modern_Art,_New_York.jpg)):
```bash
bash style/get_style_dataset.sh
```

## Train The Model
```bash
python train.py --dataset VOCdevkit/VOC2007/JPEGImages/ --style_image franc.jpg --train_epoch 100  --save_root starry_night
```

## Continue Training Existing Saved State
```bash
python train.py --dataset VOCdevkit/VOC2007/JPEGImages/ --style_image franc.jpg --train_epoch 100  --save_root starry_night  --load_state output/starry_night_3.json
```

## Command Line Arguments
```
--dataset, default='VOCdevkit/VOC2007/JPEGImages/', type=str   # Dataset folder name
--style_image, default='franc.jpg', type=str                   # Style image name
--in_channels', default=3, type=int                            # Channels in image
--batch_size, default=4, type=int                              # Training batch size
--gen_filters, default=128, type=int                           # Filter count for generator
--gen_layers, default=2, type=int                              # Count of convolution and transpose layers in generator
--res_blocks, default=5, type=int                              # Count res blocks in generator
--vgg_layers_s', default=[3, 8, 15, 22], type=int              # Layers of VGG to use for style
--vgg_layers_c', default=2, type=int                           # Layer of VGG to use for content, selected from provided style layers (vgg_layers_s[i])
--style_weight default=10000000000, type=int                   # Multiplier for style loss
--content_weight, default=10000, type=int                      # Multiplier for content loss
--lr, default=1e-3, type=float                                 # Learning rate
--train_epoch, default=60, type=int                            # Number epochs to train for
--res, default=256, type=int                                   # Image Training Size (Images is resized to this value in dataloaders)
--data_perc, default=1.0, type=float                           # Train on a subset of the training set, if set to 1 uses whole dataset
--test_perc, default=.1, type=float                            # Create test set from subset of images
--beta1, default=.5, type=float                                # Beta1 value used by optimizer
--beta2, default=.999, type=float                              # Beta2 value used by optimizer
--ids, default=[10, 20], type=int                              # Ids which will be indexed into to generate preview images from the validation set
--save_root, default='franc_style', type=str                   # Prefix for files created by the model under the /output directory
--load_state, type=str                                         # Optional: filename of state to load and resume training from
```

## Data Folder Structure

- Dataset:

`data/<data set>/`

- Style image:

`style/<style image>`

- For example if you are training `Pascal VOC 2007` training set on `Tarentelle`:

Train Dir:

`data/VOCdevkit/VOC2007/JPEGImages/`

Style Image:

`style/franc.jpg`

## Output Folder Structure

- `weights`, `test images` and `loss graph`, are all output to this directory: `output/<save_root>_*.*`

- Loss Graph Example: `output/starry_night_loss.jpg`
![](output/starry_night_loss.jpg)

- Test Image Example (output every loop through dataset): `output/starry_night_test.jpg`
![](output/starry_night_127_test.jpg)

## Other Notes

- Define a test set by setting `--test_perc` to any value above zero

- Use a small subset of the overall training by setting `--data_perc` to any value below one

