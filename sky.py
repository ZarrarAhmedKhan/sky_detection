'''
inputs:
	'image.jpg'

	Set in the given sky.py file:
		visualize = True
		save_images = "result"
		stack_images = True

*Run a file*

!python3 sky.py image.jpg

'''
import cv2
import numpy as np
import os
import sys

# canny edge detection threshold values
# '1093/20140418_215418.jpg' --> canny > 0-200
# '1093/20140421_105356.jpg' --> canny > 0-160
# '1093/20140502_165406.jpg' --> canny > 0-150

def main(input_image, visualize = False, save_image = None, stack_images = False):

	frame = cv2.imread(input_image)

	# canny edge detection
	kernel = np.ones((5,5),np.uint8)
	grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(grey_image,0,160)

	# closing --> dilation followed by erosion
	closing = cv2.morphologyEx(edges.copy(), cv2.MORPH_CLOSE, kernel)

	# dilate the edges
	# dilation = cv2.dilate(closing, kernel, iterations = 1)

	'''
		*FloodFill*
		floodflags = 4 --> only the four nearest neighbor pixels
		floodflags |= cv2.FLOODFILL_MASK_ONLY --> if need only mask image only 
		floodflags |= (255 << 8) --> 4 | ( 255 << 8 ) will consider 4 nearest neighbours and fill the mask with a value of 255.
	'''
	im_floodfill = closing.copy()
	h, w = frame.shape[:2]

	# mask shape is 2-pixels increased from width and height 
	mask = np.zeros((h+2, w+2), np.uint8)
	floodflags = 4
	# floodflags |= cv2.FLOODFILL_MASK_ONLY
	floodflags |= (255 << 8)
	_, im_floodfill,mask,rect =cv2.floodFill(im_floodfill, mask, (0,0), (50,)*3, (10,)*3, (10,)*3, floodflags)

	# convert white to black and vice-versa
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	if visualize:
		cv2.imshow("image", frame)
		cv2.waitKey(0)

		cv2.imshow("edges", edges)
		cv2.waitKey(0)

		cv2.imshow("closing", closing)
		cv2.waitKey(0)

		cv2.imshow("im_floodfill", im_floodfill)
		cv2.waitKey(0)

	cv2.imshow("output_mask", mask)
	cv2.waitKey(0)


	# resize mask shape to original image shape
	mask = cv2.resize(mask, edges.shape[::-1])

	# conunt the number of white pixels in the image(as of sky pixels)
	PixelArea = cv2.countNonZero(mask)
	print("area of sky in pixels", PixelArea)

	# find the area of the sky region
	total_pixels = mask.shape[0] * mask.shape[1]
	area_percent = (PixelArea / total_pixels) * 100
	print("sky pixels percentage: " + str(round(area_percent,2)) + '%')


	# find perimeter of sky region by using mask image
	boundary = cv2.Canny(mask, 0,150) 
	nzCount = cv2.countNonZero(boundary)
	print("perimeter of sky", nzCount)

	if visualize:	
		cv2.imshow("boundary", boundary)
		cv2.waitKey(0)

	# to get only sky region
	# inv_mask = cv2.bitwise_not(mask)
	# print(mask.shape)
	# without_sky = grey_image | inv_mask
	# cv2.imshow("without_sky", without_sky)
	# cv2.waitKey(0)

	if save_images:
		if not os.path.exists(save_images):
			os.mkdir(save_images)

		cv2.imwrite(f'{save_images}/input.jpg', frame)
		cv2.imwrite(f"{save_images}/edges.jpg", edges)
		cv2.imwrite(f"{save_images}/closing.jpg", closing)
		cv2.imwrite(f"{save_images}/floodfill.jpg", im_floodfill)
		cv2.imwrite(f"{save_images}/mask.jpg", mask)
		cv2.imwrite(f"{save_images}/boundary.jpg", boundary)
		# cv2.imwrite("result/without_sky.jpg", without_sky)

	if stack_images:
		images = os.listdir(save_images)
		h_list = []
		for i in range(len(images)):
			if i % 3 == 0:
				print(images[i-3])
				img_1 = cv2.imread(save_images + "/" + images[i-3])
				img_2 = cv2.imread(save_images + "/" + images[i-2])
				img_3 = cv2.imread(save_images + "/" + images[i-1])
				result = np.hstack((img_1, img_2, img_3))
				h_list.append(result)


		final = np.vstack((h_list[0], h_list[1]))
		cv2.imwrite("final.jpg", final)

if __name__ == '__main__':
	visualize = True
	save_images = "result"
	stack_images = True
	input_image = sys.argv[1]
	
	main(input_image, visualize, save_images, stack_images)