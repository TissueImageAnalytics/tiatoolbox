"This files computes the transformation parameters using deep feature matching method"
import os
os.add_dll_directory('F:\\Dropbox\\PhD_Work\\PythonVE\\openslide-win64-20171122\\bin')
from tiatoolbox.wsicore.wsireader import WSIReader
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from abc import ABC
import numpy as np
import cv2
from skimage import filters, morphology, measure, color
import time
from matplotlib import path

class rigidRegistration(ABC):
	def __init__(self):
		self.patch_size = (224, 224)
		self.Xscale, self.Yscale = [], []
		model = torchvision.models.vgg16(True)
		return_layers = {'16': 'block3_pool', '23': 'block4_pool', '30': 'block5_pool'}
		self.FeatureExtractor = IntermediateLayerGetter(model.features, return_layers=return_layers)

	def extract_features(self, target, source):
		# resize image
		self.Xscale = 1.0 * np.array(target.shape[:2]) / self.patch_size
		self.Yscale = 1.0 * np.array(source.shape[:2]) / self.patch_size
		fixed_cnn = cv2.resize(target, self.patch_size)
		moving_cnn = cv2.resize(source, self.patch_size)

		# input image noramlization
		fixed_cnn = fixed_cnn/255
		moving_cnn = moving_cnn/255

		# changing the channel ordering
		fixed_cnn = np.moveaxis(fixed_cnn, -1, 0)
		moving_cnn = np.moveaxis(moving_cnn, -1, 0)

		fixed_cnn = np.expand_dims(fixed_cnn, axis=0)
		moving_cnn = np.expand_dims(moving_cnn, axis=0)
		cnn_input = np.concatenate((fixed_cnn, moving_cnn), axis=0)

		x = torch.from_numpy(cnn_input).type(torch.float32)
		# x = torch.rand(2, 3, 224, 224)
		features = self.FeatureExtractor(x)
		return features

	def pairwise_distance(self, X, Y):  # for this function shape of M and N should be same
		assert len(X.shape) == len(Y.shape)
		N = X.shape[0]
		M = Y.shape[0]
		D = len(X.shape)

		return np.linalg.norm(
			np.repeat(np.expand_dims(X, axis=0), M, axis=0) -
			np.repeat(np.expand_dims(Y, axis=1), N, axis=1),
			axis=D)

	def pd_expand(self, PD, k):
		N0 = np.int(np.sqrt(PD.shape[0]))
		N1 = k * N0
		L0, L1 = N0 ** 2, N1 ** 2
		Cmat = np.kron(np.arange(L0).reshape([N0, N0]), np.ones([k, k], dtype='int32'))
		i = np.repeat(Cmat.reshape([L1, 1]), L1, axis=1)
		j = np.repeat(Cmat.reshape([1, L1]), L1, axis=0)
		return PD[i, j]

	def finding_match(self, PD):
		seq = np.arange(PD.shape[0])
		amin1 = np.argmin(PD, axis=1)
		C = np.array([seq, amin1]).transpose()
		min1 = PD[seq, amin1]
		mask = np.zeros_like(PD)
		mask[seq, amin1] = 1
		masked = np.ma.masked_array(PD, mask)
		min2 = np.amin(masked, axis=1)
		return C, np.array(min2 / min1)

	def show_matching_points(self, fixed, moving, X, Y, dist):
		keypoints1 = []
		keypoints2 = []
		matchingPoints = []
		for i in range(len(dist)):
			matchingPoints.append(cv2.DMatch(_distance=dist[i], _imgIdx=0, _queryIdx=i, _trainIdx=i))
			keypoints1.append(cv2.KeyPoint(X[i, 0], X[i, 1], 5, class_id=0))
			keypoints2.append(cv2.KeyPoint(Y[i, 0], Y[i, 1], 5, class_id=0))

		matchingPoints = sorted(matchingPoints, key=lambda x: x.distance)  # Sort them in the order of their distance.
		if np.max(fixed) <= 1:
			fixed = 255 * fixed
		fixed = fixed.astype(np.uint8)

		if np.max(moving) <= 1:
			moving = 255 * moving
		moving = moving.astype(np.uint8)
		img3 = cv2.drawMatches(fixed, keypoints1, moving, keypoints2, matchingPoints, None, flags=2)
		return img3

	def feature_mapping(self, features):
		pool3 = features['block3_pool'].detach().numpy()
		pool4 = features['block4_pool'].detach().numpy()
		pool5 = features['block5_pool'].detach().numpy()

		# flatten
		DX1, DY1 = np.reshape(pool3[0, :, :, :], [-1, 256]), np.reshape(pool3[1, :, :, :], [-1, 256])
		DX2, DY2 = np.reshape(pool4[0, :, :, :], [-1, 512]), np.reshape(pool4[1, :, :, :], [-1, 512])
		DX3, DY3 = np.reshape(pool5[0, :, :, :], [-1, 512]), np.reshape(pool5[1, :, :, :], [-1, 512])

		# normalization
		DX1, DY1 = DX1 / np.std(DX1), DY1 / np.std(DY1)
		DX2, DY2 = DX2 / np.std(DX2), DY2 / np.std(DY2)
		DX3, DY3 = DX3 / np.std(DX3), DY3 / np.std(DY3)

		# compute feature space distance
		PD1 = self.pairwise_distance(DX1, DY1)
		PD2 = self.pd_expand(self.pairwise_distance(DX2, DY2), 2)
		PD3 = self.pd_expand(self.pairwise_distance(DX3, DY3), 4)
		PD = 1.414 * PD1 + PD2 + PD3

		del DX1, DY1, DX2, DY2, DX3, DY3, PD1, PD2, PD3

		seq = np.array([[i, j] for i in range(pool3.shape[2]) for j in range(pool3.shape[2])], dtype='int32')
		X = np.array(seq, dtype='float32') * 8.0 + 4.0
		Y = np.array(seq, dtype='float32') * 8.0 + 4.0

		# normalize
		X = (X - 112.0) / 224.0
		Y = (Y - 112.0) / 224.0

		# prematch and select points
		C_all, quality = self.finding_match(PD)
		tau_max = np.max(quality)
		while np.where(quality >= tau_max)[0].shape[0] <= 128:      # 128
			tau_max -= 0.01

		C = C_all[np.where(quality >= tau_max)]
		cnt = C.shape[0]

		# select prematched feature points
		X, Y = X[C[:, 1]], Y[C[:, 0]]
		PD = PD[np.repeat(np.reshape(C[:, 1], [cnt, 1]), cnt, axis=1),
				np.repeat(np.reshape(C[:, 0], [1, cnt]), cnt, axis=0)]

		X, Y = ((X * 224.0) + 112.0) * self.Xscale, ((Y * 224.0) + 112.0) * self.Yscale
		return X, Y, np.amin(PD, axis=1)

	def remove_points_using_mask(self, mask, X, Y, dist, indx):
		kernel = np.ones((50, 50), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations=1)
		mask_X = np.array([], dtype=np.int64).reshape(0, 2)
		mask_Y = np.array([], dtype=np.int64).reshape(0, 2)
		mask_dist = np.array([], dtype=np.int64)
		bound_points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for bound_p in bound_points:
			bound_p = np.squeeze(bound_p)
			bound_p = path.Path(bound_p)

			if indx == 1:
				included_points_index = bound_p.contains_points(X)
			else:
				included_points_index = bound_p.contains_points(Y)
			tempX = X[included_points_index, :]
			tempY = Y[included_points_index, :]
			temp_dist = dist[included_points_index]

			mask_X = np.vstack([mask_X, tempX])
			mask_Y = np.vstack([mask_Y, tempY])
			mask_dist = np.hstack([mask_dist, temp_dist])
		return mask_X, mask_Y, mask_dist

	def perform_registration(self, targetI, sourceI, targetM, sourceM, removeScale=1, isDisplay=1):
		multiscale_features = self.extract_features(targetI, sourceI)
		target_points, source_points, dist = self.feature_mapping(multiscale_features)
		target_points, source_points = target_points[:, [1, 0]], source_points[:, [1, 0]]

		# remove the points which are outside the dilated mask of the fixed image
		filtered_target_points, filtered_source_points, filtered_dist = self.remove_points_using_mask(targetM, target_points, source_points, dist, 1)
		filtered_target_points, filtered_source_points, filtered_dist = self.remove_points_using_mask(sourceM, filtered_target_points, filtered_source_points, filtered_dist, 2)

		if isDisplay:
			matchingI_before = self.show_matching_points(targetI, sourceI, target_points, source_points, dist)
			matchingI_after = self.show_matching_points(targetI, sourceI, filtered_target_points, filtered_source_points, filtered_dist)

			fig = plt.figure(figsize=(60, 35))
			plt.subplot(121)
			plt.imshow(matchingI_before)
			plt.subplot(122)
			plt.imshow(matchingI_after)
			plt.show()

		transform = self.estimate_affine_transform(filtered_target_points, filtered_source_points)	# compute transformation params, if removeScale == 1, remove the scale params
		return transform

	def transform_points(self, points, matrix):
		""" transform points according to given transformation matrix

		:param ndarray points: set of points of shape (N, 2)
		:param ndarray matrix: transformation matrix of shape (3, 3)
		:return ndarray: warped points  of shape (N, 2)
		"""
		points = np.array(points)
		# Pad the data with ones, so that our transformation can do translations
		pts_pad = np.hstack([points, np.ones((points.shape[0], 1))])
		points_warp = np.dot(pts_pad, matrix.T)
		return points_warp[:, :-1]

	def estimate_affine_transform(self, points_0, points_1):
		nb = min(len(points_0), len(points_1))
		# Pad the data with ones, so that our transformation can do translations
		x = np.hstack([points_0[:nb], np.ones((nb, 1))])
		y = np.hstack([points_1[:nb], np.ones((nb, 1))])

		# Solve the least squares problem X * A = Y to find our transform. matrix A
		matrix = np.linalg.lstsq(x, y, rcond=-1)[0].T
		matrix[-1, :] = [0, 0, 1]

		# # invert the transformation matrix
		# matrix_inv = np.linalg.pinv(matrix.T).T
		# matrix_inv[-1, :] = [0, 0, 1]
		#
		# points_0_warp = self.transform_points(points_0, matrix)
		# points_1_warp = self.transform_points(points_1, matrix_inv)
		#
		# return matrix, matrix_inv, points_0_warp, points_1_warp
		return matrix

def normalize(image):
	# percentiles = np.percentile(image, (1, 99))
	# scaled = exposure.rescale_intensity(image, in_range=tuple(percentiles))/255
	scaled = (image - np.min(image)) / (np.max(image) - np.min(image))
	return scaled

def preprocess(target, source, echo=True):
	def image_entropy(image):
		return filters.rank.entropy(image, morphology.disk(3))

	def histogram_correction(source, target):
		oldshape = source.shape
		source = source.ravel()
		target = target.ravel()

		s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
		t_values, t_counts = np.unique(target, return_counts=True)

		sq = np.cumsum(s_counts).astype(np.float64)
		sq /= sq[-1]
		tq = np.cumsum(t_counts).astype(np.float64)
		tq /= tq[-1]
		interp_t_values = np.interp(sq, tq, t_values)
		return interp_t_values[bin_idx].reshape(oldshape)

	b_time_hist = time.time()
	if len(source.shape) == 3:
		source = color.rgb2gray(source)
	if len(target.shape) == 3:
		target = color.rgb2gray(target)

	source, target = normalize(source), normalize(target)
	source_entropy, target_entropy = image_entropy(source), image_entropy(target)

	if echo:
		print("Source entropy: ", np.mean(source_entropy))
		print("Target entropy: ", np.mean(target_entropy))

	if np.mean(target_entropy) > np.mean(source_entropy):
		source = histogram_correction(source, target)
	else:
		target = histogram_correction(target, source)
	e_time_hist = time.time()

	if echo:
		print("Time for histogram correction: ", e_time_hist - b_time_hist, " seconds.")

	source, target = normalize(source), normalize(target)
	target = np.repeat(target[:, :, np.newaxis], 3, axis=2)
	source = np.repeat(source[:, :, np.newaxis], 3, axis=2)
	return target, source

def simple_get_mask(rgb):
	gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) # threshold using the OTSU method
	mask = morphology.remove_small_objects(mask == 0, min_size=100, connectivity=2)
	mask = morphology.remove_small_holes(mask, area_threshold=100)
	mask = morphology.binary_dilation(mask, morphology.disk(5))
	return mask

def get_mask(grayscale):
	# grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	ret, mask = cv2.threshold(grayscale, np.mean(grayscale), 255, cv2.THRESH_BINARY)
	mask = morphology.remove_small_objects(mask == 0, min_size=1000, connectivity=2)
	mask = morphology.binary_opening(mask, morphology.disk(3))
	mask = morphology.remove_small_objects(mask == 1, min_size=1000, connectivity=2)
	mask = morphology.remove_small_holes(mask, area_threshold=100)

	# remove all the objects while keep the biggest object only
	label_img = measure.label(mask)
	regions = measure.regionprops(label_img)
	mask = mask.astype(bool)
	all_area = [i.area for i in regions]
	second_max = max([i for i in all_area if i != max(all_area)])
	mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
	mask = mask.astype(np.uint8)
	return mask

def main():
	# Setting paths to target and source WSIs
	data_dir = 'G:\\WSI_Registration_Dataset\\Multi-stain_Dataset\\UHCW\\Omnyx\\MEVIS\\Case1'
	target_wsi_name = 'Case1_Section1.tif'
	source_wsi_name = 'Case1_Section2.tif'
	target_wsi_path = '%s/%s' % (data_dir, target_wsi_name)
	source_wsi_path = '%s/%s' % (data_dir, source_wsi_name)

	# reading downsampled version of target and source WSIs
	target_wsi_reader = WSIReader.open(input_img=target_wsi_path)
	target_wsi_thumb = target_wsi_reader.slide_thumbnail(resolution=0.3125, units='power')
	source_wsi_reader = WSIReader.open(input_img=source_wsi_path)
	source_wsi_thumb = source_wsi_reader.slide_thumbnail(resolution=0.3125, units='power')

	# image preprocessing
	[targetI, sourceI] = preprocess(target_wsi_thumb, source_wsi_thumb, echo=True)

	# generate tissue masks
	targetM = get_mask(targetI[:,:,0])
	sourceM = get_mask(sourceI[:,:,0])

	# displaying thumbnails
	# plt.subplot(221)
	# plt.imshow(targetI, cmap='gray')
	# plt.subplot(222)
	# plt.imshow(sourceI, cmap='gray')
	# plt.subplot(223)
	# plt.imshow(targetM)
	# plt.subplot(224)
	# plt.imshow(sourceM)
	# plt.axis('off')
	# plt.show()

	# DFBR based rigid registration
	reg = rigidRegistration()
	transform = reg.perform_registration(targetI, sourceI, targetM, sourceM)
	registeredI = cv2.warpAffine(sourceI, transform[0:-1][:], targetI.shape[:2][::-1])

	plt.imshow(targetI)
	plt.imshow(registeredI, alpha=0.5)
	plt.show()
	print('')

if __name__ == "__main__":
	main()


