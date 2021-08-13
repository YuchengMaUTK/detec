import numpy as np
import cv2

def blend(im1, im2, mask):
  mask = mask / 255.
  im1 = im1/255.
  im2 = im2/255.
  #print(im1.shape,im2.shape, mask.shape)
  dim = (300, 300)
  im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
  py_level = 5

  #Build Laplacian pyramids LA and LB from images A and B
  _, laplace1 = createPyramid(im1, py_level)
  _, laplace2 = createPyramid(im2, py_level)
  gaussianMask, _ = createPyramid(mask, py_level+1)

  laplaceMerge = []

  #Form a combined pyramid LS from LA and LB
  #LS(i, j) = GR(I, j, ) * LA(I, j) + (1 - GR(I, j)) * LB(I, j)
  for i in range(py_level):
    one_array = np.ones(gaussianMask[i].shape)
    laplaceMerge.append(laplace1[i] * gaussianMask[i] + laplace2[i] * (one_array - gaussianMask[i]))


  imagemerged = gaussianMask[-1]

  for i in range(py_level):
    imagemerged = cv2.resize(imagemerged, (laplaceMerge[-i - 1].shape[1], laplaceMerge[-i - 1].shape[0]))
    imagemerged = imagemerged + laplaceMerge[-i - 1]

  imagemerged = im1*imagemerged +  im2*(np.ones(imagemerged.shape) - imagemerged)
  out = imagemerged*255.
  return out



def createPyramid(im, level):
  gaussianPyramid,laplacePyramid = [],[]

  for i in range(level):
    newstack = cv2.resize(im, (0, 0), fx=1/2, fy=1/2)
    #print(newimg.shape)
    newgaussian = cv2.resize(im, (im.shape[1], im.shape[0]))
    #print(newgaussian.shape)
    gaussianPyramid.append(newgaussian)

    laplacePyramid.append(im - newgaussian)
    im = newstack

  return gaussianPyramid, laplacePyramid