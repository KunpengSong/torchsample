# from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def time_left_str(seconds):
    # seconds = 370000.0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        thetime = "Projected time remaining  |  {:d}d:{:d}h:{:02d}m".format(d, h, m)
    elif h > 0:
        thetime = "Projected time remaining:  |  {:d}h:{:02d}m".format(h, m)
    elif m > 0:
        thetime = "Projected time remaining:  |  {:02d}m:{:02d}s".format(m, s)
    else:
        thetime = "Projected time remaining:  |  {:02d}s".format(seconds)
    return thetime