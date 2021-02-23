import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
#from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np
# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
import cv2
import PIL
from PIL import Image
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import PIL
from PIL import Image
import torch.backends.cudnn as cudnn
import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB
from numpy import array

import io
# get the image path

from flask import  Flask, request, Response, jsonify, request,render_template, send_from_directory,Response
from werkzeug.utils import secure_filename
from models.experimental import *
from utils.datasets import *
from utils import *
import sys
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords,xyxy2xywh, plot_one_box, strip_optimizer, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
import os,logging
import platform
import shutil
import time, io
from pathlib import Path
from torchvision.utils import save_image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import cv2

from numpy import asarray

app = Flask(__name__)
weights = "weights/best6.pt"
device_number = '' 
device = torch_utils.select_device(device_number)

model = attempt_load(weights, map_location=device)  # load FP32 model
out = "inference/output"
crop ="inference/crop"
global text

def detect():
	save_img=False
	form_data = request.json
	print(form_data)

	source = "inference/images/"
	out = "inference/output"
	#ut = form_data['output']
	imgsz = 640
	conf_thres = 0.5
	iou_thres = 0.5
	view_img = False
	save_txt = False
	classes = None
	agnostic_nms = False
	augment = False
	update = False
	save_img=False
	global imageafterpred, bound
	#form_data = request.json
	#print(form_data)
	bound = []
	

	webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    
    # Initialize
    # device = torch_utils.select_device(opt.device)
	if os.path.exists(out):
		shutil.rmtree(out)  # delete output folder
	os.makedirs(out)  # make new output folder
	half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	if half:
		model.half()  # to FP16

    # Second-stage classifier
	classify = False
	if classify:
		modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		modelc.to(device).eval()

    # Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = True
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz)
	else:
		save_img = True
		dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)
			save_image(img, 'temp.png')

        # Inference
		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=augment)[0]

        # Apply NMS
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
		t2 = torch_utils.time_synchronized()

        # Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			else:
				p, s, im0 = path, '', im0s
				imageafterpred = im0

			save_path = str(Path(out) / Path(p).name)
			txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if det is not None and len(det):
			# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
				for c in det[:, -1].detach().unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
				for *xyxy, conf, cls in det:
					xyx =torch.tensor(xyxy).view(1, 4)
					xyx = xyx.numpy()
					#print(xyx)
					x1,y1,x2,y2= xyx[0][0],xyx[0][1],xyx[0][2],xyx[0][3]
					#print(x1,y1,x2,y2)
					bound.append(x1)
					bound.append(y1)
					bound.append(x2)
					bound.append(y2)
					#print(bound)
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						print(xywh)
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

					if save_img or view_img:  # Add bbox to image
						label = '%s %.2f' % (names[int(cls)], conf)
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
			print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
			if view_img:
				cv2.imshow(p, im0)
				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration

            # Save results (image with detections)
			if save_img:
				if dataset.mode == 'images':
					cv2.imwrite(save_path, im0)
				else:
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer
						fourcc = 'mp4v'  # output video codec
						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
					vid_writer.write(im0)

	if save_txt or save_img:
		print('Results saved to %s' % os.getcwd() + os.sep + out)
		if platform == 'darwin' and not update:  # MacOS
			os.system('open ' + save_path)

	print('Done. (%.3fs)' % (time.time() - t0))
	return imageafterpred ,bound,out
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
model1.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")


app = Flask(__name__)
run_with_ngrok(app)





# route http posts to this method
global img_name,image
@app.route('/test',methods=['GET', 'POST'])
def test():
		res, bound,path = detect()
		x1,y1,x2,y2 = bound[0],bound[1],bound[2],bound[3]
		
		for i in glob.glob("inference/images/*.jpg"):
			path = str(i)
			image2 = Image.open(path)
			image2.save("temp.png")
		image2 = Image.open('temp.png')
		im1 = image2.crop((x1,y1,x2,y2))
		wid,height = im1.size
		if wid> 1000 and height >500:
			im1.resize((700,400))
		if wid < 1000 and height > 500:
			im1.resize((700,400))
		if wid > 700 and height <500:
			im1.resize((700,height))

		print(wid,height)
		text = "finished"
       
		return text
			



if __name__ == '__main__':
  app.run()
