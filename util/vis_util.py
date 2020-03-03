from math import sqrt,ceil
import numpy as np

def visual_grid(Xs,ubound = 255.0 ,padding =1 ):
    """
    return a grid of a image by reshape this grid
    :param Xs: Data of shape (N,H,W,C)
    :param ubound: output grid will have values scaled to the range(0,unbound)
    :param padding:the number of blank between elements of the grid
    :return:
    """
    (N,H,W,C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size-1)
    grid_width = W * grid_size + padding * (grid_size-1)
    grid = np.zeros((grid_height,grid_width,C))
    next_idx = 0

    y0 ,y1 = 0,H

    # fill the all grid . row first, column second .
    for y in range(grid_size):
        x0,x1  = 0,W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low,high = np.min(img),np.max(img)
                grid[x0:x1,y0:y1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding

        y0 += H + padding
        y1 += H + padding

    return grid


def vis_nn(rows):
    """
    visual arrays of arrays image
    :param rows:
    :return:
    """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_grid(Xs):
    """
    visual gird of images
    :param Xs: images
    :return:
    """
    (N,H,W,C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A,A*W+A,C),Xs.dtype)
    G *= np.min(Xs)
    n = 0

    # you know the y is the images number and y is also the padding
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y,x*W+x:(x+1)*W+x]  = Xs[n,:,:,:]
                n += 1
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G


