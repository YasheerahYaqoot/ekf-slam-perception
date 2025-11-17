import numpy as np
from PIL import Image, ImageDraw

def convert2xy(scan, fov=360, min_dist=0.002):
    angles = np.radians(np.linspace(-fov/2, fov/2, len(scan)))
    points = np.vstack([scan*np.cos(angles), scan*np.sin(angles)]).T
    return points[scan>min_dist]

def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    x = points[:, 0]*c - points[:, 1]*s
    y = points[:, 0]*s + points[:, 1]*c
    return np.vstack([x, y]).T

def convert2map(pose, points, map_pix, map_size, prob):
    zero = (pose//map_pix).astype(np.int32)
    pixels = (points//map_pix).astype(np.int32)
    mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < map_size[0]) & \
           (pixels[:, 1] >= 0) & (pixels[:, 1] < map_size[1])
    pixels = pixels[mask]
    img = Image.new('L', (map_size[1], map_size[0]))
    draw = ImageDraw.Draw(img)
    zero = (zero[1], zero[0])
    for p in set([(q[1], q[0]) for q in pixels]):
        draw.line([zero, p], fill=1)
    data = -np.fromstring(img.tobytes(), np.int8).reshape(map_size)
    data[pixels[:, 0], pixels[:, 1]] = 1
    return 0.5 + prob*data.astype(np.float32)

