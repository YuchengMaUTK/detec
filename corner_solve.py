import numpy as np
import scipy
import cv2

def compute_corners(I):
  # Currently this code proudces a dummy corners and a dummy corner response
  # map, just to illustrate how the code works. Your task will be to fill this
  # in with code that actually implements the Harris corner detector. You
  # should return th ecorner response map, and the non-max suppressed corners.
  # Input:
  #   I: input image, H x W x 3 BGR image
  # Output:
  #   response: H x W response map in uint8 format
  #   corners: H x W map in uint8 format _after_ non-max suppression. Each
  #   pixel stores the score for being a corner. Non-max suppressed pixels
  #   should have a low / zero-score.
  
  rng = np.random.RandomState(int(np.sum(I.astype(np.float32))))
  sz = I.shape[:2]
  corners = rng.rand(*sz)
  response = corners
  I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I_gray = I_gray.astype(np.float32) / 255.
  #print(I_gray)

  dx = scipy.signal.convolve2d(I_gray, np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]), boundary='symm', mode='same')
  dy = scipy.signal.convolve2d(I_gray, np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).T, boundary='symm', mode='same')
  Ixx = dx ** 2
  Ixy = dy * dx
  Iyy = dy ** 2

  height, weight = I_gray.shape
  offset = 1
  sigma = 0.8
  a = 0.03

  #compute response
  for i in range(offset, height - offset):
    for j in range(offset, weight - offset):
      window_x = scipy.signal.windows.gaussian((3), std=sigma)
      window_y = scipy.signal.windows.gaussian((3), std=sigma)
      window = np.outer(window_x, window_y)
      S_xx = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1]
      S_yy = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1]
      S_xy = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1]
      S_xx = np.sum(window * S_xx)
      S_yy = np.sum(window * S_yy)
      S_xy = np.sum(window * S_xy)

      det = (S_xx * S_yy) - (S_xy ** 2)
      trace = S_xx + S_yy
      response[i, j] = det - a * trace ** 2

  det_x0 = Ixx[0]*Iyy[0] - (Ixy[0]**2)
  trace_x0 = Ixx[0] +Iyy[0]
  response[0] = det_x0 - a*trace_x0**2
  det_xm1 =  Ixx[-1]*Iyy[-1] - (Ixy[-1]**2)
  trace_xm1 = Ixx[-1] + Iyy[-1]
  response[-1] = det_xm1 - a * trace_xm1 ** 2
  det_y0 = Ixx[:,0] * Iyy[:,0] - (Ixy[:,0] ** 2)
  trace_y0 = Ixx[:,0] + Iyy[:,0]
  response[:,0] = det_y0 - a * trace_y0 ** 2
  det_ym1 =  Ixx[:,-1]*Iyy[:,-1] - (Ixy[:,-1]**2)
  trace_ym1 = Ixx[:,-1] + Iyy[:,-1]
  response[:,-1] = det_ym1 - a * trace_ym1 ** 2

  # for i in range(offset, height - offset):
  #   for j in range(offset, weight - offset):
  #     S_xx = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1]
  #     S_yy = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1]
  #     S_xy = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1]
  #     S_xx = np.sum(S_xx)
  #     S_yy = np.sum(S_yy)
  #     S_xy = np.sum(S_xy)
  #
  #     det = (S_xx * S_yy) - (S_xy ** 2)
  #     trace = S_xx + S_yy
  #     response[i, j] = det - a * trace ** 2
  threshold = 0.0000001
  adaptive_threshold = threshold * np.max(response)

  #NMS
  offset = 1
  newcorners = np.zeros((height, weight), dtype=corners.dtype)
  for i in range(offset, height - offset):
    for j in range(offset, weight - offset):
      window = response[i - offset:i + offset + 1, j - offset:j + offset + 1]
      peak = np.amax(window)
      if response[i, j] == peak and response[i, j]>adaptive_threshold:
        newcorners[i, j] = peak

  corners = newcorners

  corners = corners * 255.
  corners = np.clip(corners, 0, 255)
  corners = corners.astype(np.uint8)

  response = response * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)


  
  return response, corners
