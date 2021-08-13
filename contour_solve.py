import numpy as np
from scipy import signal
import cv2

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  #I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Gaussian filter
  I = I.astype(np.float32) / 255.
  sigma = 2
  size = 9

  x_points = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
  y_points = x_points[::-1]
  xs, ys = np.meshgrid(x_points, y_points)
  kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (xs ** 2 + ys** 2) / (2 * sigma ** 2)) * (-ys / sigma ** 2)
  kernel = kernel/kernel.sum()

  dx = np.zeros(I.shape)
  dy = np.zeros(I.shape)
  mag = np.zeros(I.shape)
  for channel in range(3):
    dx[:, :, channel] = signal.convolve2d(I[:, :, channel], kernel, mode='same', boundary='symm')
    dy[:, :, channel] = signal.convolve2d(I[:, :, channel], kernel.T, mode='same', boundary='symm')
    mag[:, :, channel] = np.sqrt(dx[:, :, channel] ** 2 + dy[:, :, channel] ** 2)
    mag[:, :, channel] = mag[:, :, channel] / np.max(mag[:, :, channel])
  # dx = signal.convolve2d(I, kernel, mode='same', boundary='symm')
  # dy = signal.convolve2d(I, kernel.T, mode='same', boundary='symm')
  mag = np.sqrt(np.sum(mag ** 2, axis=2))
  #mag = np.sqrt(dx ** 2 + dy ** 2)
  mag = mag / np.max(mag)

  dx = dx.sum(axis=2)
  dy = dy.sum(axis=2)

  #Non-maximum suppression
  height,width= mag.shape
  for i in range(1, height - 1):
    for j in range(1, width - 1):
      grade = np.sqrt(dx[i][j] ** 2 + dy[i][j] ** 2)
      if mag[i][j] <= 0 or mag_check(mag, [i + 1 * dx[i][j] / grade, j + 1 * dx[i][j] / grade], [i, j]) or mag_check(mag, [i - 1 * dx[i][j] / grade, j - 1 * dx[i][j] / grade], [i, j]):
        mag[i][j] = 0


  mag = mag * 255
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag

def mag_check(mag, xy, cur):
  #Bilinear Interpolation
  F = np.array([[mag[int(xy[0]), int(xy[1])], mag[int(xy[0]), int(np.ceil(xy[1]))]],[mag[int(np.ceil(xy[0])), int(xy[1])], mag[int(np.ceil(xy[0])), int(np.ceil(xy[1]))]]])
  x = xy[0] - int(xy[0])
  y = xy[1] - int(xy[1])
  mag_inter = np.array([[1 - x, x]]).dot(F.dot(np.array([[1 - y, y]]).T))
  if mag_inter >= mag[cur[0], cur[1]]: return True
  else: return False
