import os
import keras
import numpy as np
import scipy.signal
from skimage import io, exposure, img_as_uint, img_as_float
import scipy.misc
from scipy.misc import imsave
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter
import math
import glob
from PIL import Image
from decimal import * 
import cv2
import sys
import numexpr as ne
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#-------------load and display functions-----------------


def save_feature(feature, image_list, i, j):

	im = np.array(feature, dtype='float64')
	im = exposure.rescale_intensity(im, out_range='float')
	im = img_as_uint(im)

	image_list.append(im)

def load_model(model_name, image_shape, include_top=True):
	# Load wanted model (resnet, vgg16 or vgg19)

	# set params
	weights = 'imagenet'
	model = None

	if model_name == 'resnet':
		model = keras.applications.resnet50.ResNet50(
			include_top=include_top,
			weights=weights,
			input_shape=image_shape
		)
	elif model_name == 'vgg16':
		model = keras.applications.vgg16.VGG16(
			include_top=include_top,
			weights=weights,
			input_shape=image_shape
		)
	elif model_name == 'vgg19':
		model = keras.applications.vgg19.VGG19(
			include_top=include_top,
			weights=weights,
			input_shape=image_shape
		)
	else:
		print("Model name is unknown")
	return model


def show_layers(model):

	# Show all layers of model
	print("Layers :")
	for layer in model.layers:
		print("  %s\t--> %s" % (layer.name, layer.output.shape))

def load_image(image_name, image_shape=None):

	image = cv2.imread(image_name).astype(np.float32)
	if image_shape != None:
		image = cv2.resize(image, image_shape)
	# Remove train image mean (imagenet)
	image[:,:,0] -= 103.939
	image[:,:,1] -= 116.779
	image[:,:,2] -= 123.68
	image = np.expand_dims(image, axis=0)
	return image


#-------------features exxtraction function-----------------


def extract_features(model, image, n, feature_list, num_layer):

	# Try to extract every layer features
	from keras import backend as K
	inp = model.input						 # input placeholder
	outputs = [layer.output for layer in model.layers]	# all layer outputs
	functor = K.function([inp]+[K.learning_phase()], outputs )  # evaluation function
	# Forward the network
	layer_outs = functor([image, 1.])
	for i,features in enumerate(layer_outs[int(num_layer):]): #select 2 last layers
		
		features = features[0] 
		f_range = range(0, features.shape[2])

		for j in f_range: #iterate on the features of the layer
				
			image = features[:,:,j]
			save_feature(image, feature_list, i, j)
		

#------------------saliency methods---------------------

def bilinear_rescale(feature_list):

	feature_list_resized = []
	
	width, heigth = feature_list[len(feature_list)/2].shape
	final_size = [width,heigth]

	new_im = Image.new('RGB', (final_size[0], final_size[1]))
	
	i = 0

	for f in feature_list:
		new_im = scipy.misc.imresize(f, (final_size[0],final_size[1]), 'bilinear')
		feature_list_resized.append(new_im)
		i += 1

	return feature_list_resized
			
			
def linear_combination(feature_list):
	
	weight = 1/float(len(feature_list))
	print(len(feature_list))
	i = 0

	for feature in feature_list:

		if i == 0:
			img = feature 
			final_image = ne.evaluate('img * weight')
			i += 1

		if i < len(feature_list):
			img = feature 
			final_image = final_image + ne.evaluate('img * weight')
			i+=1

	return final_image 

def softmax(im):

	im = Image.open(im)

	pixels = list(im.getdata())
	width, height = im.size
	pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]

	sum = 0

	final = pixels

	#normalize by standard deviation
	std_pixel = pixels / np.std(pixels)

	#sum of exp
	for p in std_pixel:
		for n in p:
			sum += math.exp(n)

	#final array
	i = 0
	for p in std_pixel:
		j=0
		for n in p:
			final[i][j]=math.exp(n)/sum
			j += 1
		i += 1

	max = 0 

	i = 0
	for p in final:
		j=0
		for n in p:
			if(n>max):
				max = n
			j += 1
		i += 1

	i = 0
	for p in std_pixel:
		j=0
		for n in p:
			final[i][j]=(final[i][j]/max)*255
			j += 1
		i += 1

	array = np.array(final, dtype=np.uint8)
	new_image = Image.fromarray(array)

	return new_image


def bias(img,width,heigth):

	bias = scipy.misc.imread('bias.png','L')

	final_size = [width,heigth]
	bias = scipy.misc.imresize(bias,(final_size[0],final_size[1]), 'bilinear')
	
	img = img * bias 
	return img



def main(argv):

	start = timer()

	sig = 3
	layer = 1
	m = "vgg16"
	input_image = ""

	if "-g" in argv:
		sig = argv[argv.index("-g")+1]

	if "-l" in argv:
		layer = argv[argv.index("-l")+1]

	if "-m" in argv:
		m = argv[argv.index("-m")+1]

	if "-i" in argv:
		input_image = argv[argv.index("-i")+1]

	image = load_image(input_image)
	model = load_model(m,image.shape[1:4],False)

	feature_list = []

	print('feature extraction')
	extract_features(model,image,0, feature_list, layer)
	print('done')
	print('-----------------')

	width, heigth = image.shape[1:3]
	final_size = [width,heigth]

	width_bis, heigth_bis = feature_list[len(feature_list)/2].shape
	print len(feature_list)

	print('feature rescaling')
	feature_list = bilinear_rescale(feature_list)
	print('done')
	print('-----------------')
	print('uniform linear combination')
	final = linear_combination(feature_list)
	#scipy.misc.imsave('out/combined.jpg',final)
	feature_list = []
	print('done')
	print('-----------------')
	print('gaussian filtering')
	final = gaussian_filter(final,sigma=int(sig))
	#scipy.misc.imsave('out/filtered.jpg',final)
	print('done')
	print('-----------------')
	print('center bias applying')
	final = bias(final,width_bis,heigth_bis)
	scipy.misc.imsave('out/bias_map.jpg',final)
	print('done')
	print('-----------------')
	print('softmax funciton')		
	final = softmax('out/bias_map.jpg')
	smap = scipy.misc.imresize(final, (width,heigth), 'bilinear')
	end = timer()
	time = end - start
	scipy.misc.imsave('out/selection3/'+(input_image.split('.')[0]).split('/')[-1]+'_sig'+str(sig)+'_layer'+str(layer)+'_'+m+'_'+str(time)+'secmap.jpg',smap)
	#scipy.misc.imsave('out/BenchmarkIMAGES/'+input_image.split('/')[-1],smap)
	os.system('rm -rf out/bias_map.jpg')
	print('done')
	print('-----------------')
	print('Saliency map computed in '+str(time)+' sec')	
	print('-----------------')
	print sig, layer, m, input_image

if __name__ == "__main__":
	main(sys.argv)
