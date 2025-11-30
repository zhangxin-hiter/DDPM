from PIL import Image
import numpy

def keep_image_size_open(path, size=(256, 256)):

    VOC_COLORMAP = [
    [0, 0, 0],          # background / void
    [128, 0, 0],        # aeroplane
    [0, 128, 0],        # bicycle
    [128, 128, 0],      # bird
    [0, 0, 128],        # boat
    [128, 0, 128],      # bottle
    [0, 128, 128],      # bus
    [128, 128, 128],    # car
    [64, 0, 0],         # cat
    [192, 0, 0],        # chair
    [64, 128, 0],       # cow
    [192, 128, 0],      # diningtable
    [64, 0, 128],       # dog
    [192, 0, 128],      # horse
    [64, 128, 128],     # motorbike
    [192, 128, 128],    # person
    [0, 64, 0],         # potted plant
    [128, 64, 0],       # sheep
    [0, 192, 0],        # sofa
    [128, 192, 0],      # train
    [0, 64, 128]        # tv/monitor
    ]

    palette = []

    for i in range(256):
        if i < len(VOC_COLORMAP):
            palette.extend(VOC_COLORMAP[i])
        else:
            palette.extend([0, 0, 0])

    img = Image.open(path)
    img.putpalette(palette)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.putpalette(palette)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

if __name__ == "__main__":
    print(keep_image_size_open("assets/VOC2012/SegmentationClass/2011_003256.png").save("test.png"))