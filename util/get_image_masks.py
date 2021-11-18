import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import measure
import argparse

def get_image_masks(img):
	img_masks = []
	values = np.unique(img)
	values = values[1:] #get rid of 0
	for v in values:
		img_filtered = np.zeros(img.shape)
		img_filtered[np.where(img == v)] = v
		labels = measure.label(img_filtered, background=0)
		blobs = np.unique(labels)
		blobs = blobs[1:]
		if len(blobs) == 1:
			img_masks.append(img_filtered)
		else:
			for b in blobs:
				img_filtered = np.zeros(img.shape)
				img_filtered[np.where(labels == b)] = v
				img_masks.append(img_filtered)
	return np.array(img_masks)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help = "input image")
	args = parser.parse_args()
	if args.input is None:
		print("You must supply an input image.")
		exit()

	img = io.imread(args.input)
	img_masks = get_image_masks(img)
	for img in img_masks:
		print(np.unique(img)[1:])
		plt.imshow(img)
		plt.show()
