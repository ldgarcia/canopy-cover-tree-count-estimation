# Segmentation configurations (shown using the Rwanda dataset)

## UNet A1
```bash
python3 ${DLC_PROJECT_DIRECTORY}/scripts/train_eval.py --config \
    segmentation/unet/a1 \
    ds/rwanda \
    --train 400 --seed 0
```

## VGG16-UNet A1
```bash
python3 ${DLC_PROJECT_DIRECTORY}/scripts/train_eval.py --config \
    segmentation/vgg16_unet/a1 \
    ds/rwanda \
    --train 400 --seed 0
```

## VGG16-UNet A2
```bash
python3 ${DLC_PROJECT_DIRECTORY}/scripts/train_eval.py --config \
    segmentation/vgg16_unet/a2 \
    ds/rwanda \
    --train 400 --seed 0
```
