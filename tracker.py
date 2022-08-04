import numpy as np

import cv2 as cv

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_laplace as gf

import tqdm

import sys
import multiprocessing
from multiprocessing import Pool

FNAME = sys.argv[1]
VID = cv.VideoCapture(FNAME)
orig_xdim, orig_ydim = int(VID.get(cv.CAP_PROP_FRAME_WIDTH)), int(VID.get(cv.CAP_PROP_FRAME_HEIGHT))
a_r = orig_ydim/orig_xdim  # aspect ratio
global XDIM
global YDIM
downscale = True
if downscale:
    XDIM, YDIM = 512, int(a_r*orig_xdim)
else:
    XDIM, YDIM = orig_xdim, orig_ydim

    
FRAME_COUNT = int(VID.get(cv.CAP_PROP_FRAME_COUNT))
FPS = VID.get(cv.CAP_PROP_FPS)


params = cv.SimpleBlobDetector_Params()
params.filterByColor = False
params.minRepeatability = 3
blob = cv.SimpleBlobDetector_create(params)



## HELPER FUNCTIONS
def to_uint8(img):
    """ Rescales float image to unint8 for openCV Blob detector """
    if img.dtype == 'uint8':
        return img
    
    img -= img.min()
    return (255*img/img.max()).astype('uint8')

def rand_rgb():
    """ Generate randint rgb int for opencv"""
    return tuple([int(np.random.randint(100, 256, 1)) for _ in range(3)])

def draw_coords(img, coords, colour):
    for c in coords:
        img = cv.drawMarker(img, (int(c.pt[0]), int(c.pt[1])), colour, markerSize=30)
    return img

def d(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    return np.linalg.norm(x1 - x2)

def nearest(init, coords):
    best_dist = 1e7
    c2 = None
    for c in coords:
        ds = d(c.pt, init.pt)
        if ds < best_dist:
            best_dist = ds
            c2 = c
            
    return c2

def write_batch(frames, writer):
    for f in frames:
        writer.write(f)

def num_points(x):  # 1st argument is img and second is scale
    img_u = to_uint8(gf(x[0], x[1]))
    return len(blob.detect(img_u))


def vid2array(n_frames):
    """ Takes n_frames frames from video stream and converts
    to numpy array of dimensions (n_frames, YDIM, XDIM) """
    array = np.empty((n_frames, YDIM, XDIM), dtype="uint8")

    for i in range(n_frames):
        _, frame = VID.read()  ## put assert dim here just in case
        array[i] = cv.cvtColor(cv.resize(frame, (XDIM, YDIM)), cv.COLOR_BGR2GRAY)
        
    return array


def find_scale(img, t0=0, t1=7):
    n_points = []; T = np.linspace(t0, t1, 20)
   
    with Pool(multiprocessing.cpu_count()) as p:
        n_points = p.map(num_points, list(zip([img]*20, T)))

    x1 = np.argmax(n_points)
    x2 = np.argmin(n_points[x1:]) + x1
   
    return T[x2]


def compute_write_coords(n_frames, writer, init_coords=None):
    """ We take a batch of n_frames, compute the cooordinates, 
    find the particle paths in this batch, draw the paths and write the image to
    a .mp4 file. We then return the last frame's coords to use for the next 
    batch. By passing dummy_array, we save time creating a new array every time"""
    
    frames = vid2array(n_frames)
    bframes = frames - np.mean(frames, axis=0)  # background subtraction

    scale = find_scale(bframes[0])  # We only find scale for at the start of each batch

    coords = []  # This wound tuples of integers but opencv coordinate objects themselves

    new = np.empty((*frames.shape, 3)).astype('uint8')  #extra dim for rgb image
    for i in range(len(frames)):
        points = blob.detect(to_uint8(gf(bframes[i], scale)))
        rgb_frame = np.repeat(np.expand_dims(to_uint8(frames[i]), axis=2), 3, 2)
        new[i] = rgb_frame

        coords.append(points)

    paths = track_ps_t(init_coords, coords)  # FOR NEW PATH CREATING FUNCTIONS, SUB IT IN FOR THIS FUNC

    for p in paths:
        new = draw_path_vid(new, p, rand_rgb())

    write_batch(new, writer)
    return coords


def track_ps_t(init_coords, coords, lookahead=3):
    """ Pretty rudimentary path finding method. Simply takes looks for the
    nearest coordinte in the next n 'lookahead' frames for each coordinate for a frames. 
    The initial points of the  paths will always come from the first index of 
    coords, so how many  paths calulated will be len(coords[0]). """
    paths = []
    if init_coords != None:
        coords = [init_coords, *coords]
        
    for c_init in coords[0]:
        chain = []
        ct = c_init
        for t in range(1, len(coords)-lookahead):
            nextn = []
            for l in range(lookahead):
                nextn += coords[t+l]
                
            if len(nextn) == 0:
                chain.append([ct, ct])
                continue
            
            ctn = nearest(ct, nextn)
            chain.append([ct, ctn])
            ct = ctn
        paths.append(chain)

    return paths


def draw_line(img, c1, c2, colour): 
    c1 = (int(c1.pt[0]),int(c1.pt[1])); c2 = (int(c2.pt[0]),int(c2.pt[1]))
    return cv.line(img, c1, c2, colour)


def draw_path(img, chain, colour=rand_rgb()):
    img = draw_coords(img, [chain[0][0]], colour)
    for c in chain:
        img = draw_line(img, c[0], c[1], colour)
    return img 


def draw_path_vid(vid, chain, colour=rand_rgb()):
    img = draw_coords(vid[0], [chain[0][0]], colour)
    for t in range(1, len(chain)): 
        vid[t] = draw_coords(vid[t], [chain[t-1][0]], colour) 
        vid[t] = draw_path(vid[t], chain[:t], colour)
    return vid


def main():
    cur_frame = 0
    batch_size = 40  # how much frames we process at once
    total_batches = max(1, int(FRAME_COUNT/batch_size))
    # total_batches = 4
    codec = cv.VideoWriter_fourcc(*'MJPG')
    new_fname = FNAME + '_TRACKED.mp4'
    writer = cv.VideoWriter(new_fname, codec, FPS, (XDIM, YDIM))

    cur_coords = None
    try:
        for i in tqdm.tqdm(range(total_batches)):
            if (FRAME_COUNT - cur_frame) <= 2*batch_size:
                batch_size = FRAME_COUNT - cur_frame
                l = compute_write_coords(batch_size-1, writer, cur_coords)
                break

            else:
                cur_coords = compute_write_coords(batch_size-1, writer, cur_coords)
                
        writer.release()
        print("Wrote new file to ", new_fname)

    except KeyboardInterrupt:
        print("Stopping early...")
        writer.release()
        print("Wrote new file to ", new_fname)


if __name__=='__main__':
    main()
