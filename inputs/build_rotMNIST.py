from sklearn.datasets import fetch_mldata
import numpy as np
import cv2
import os
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import *

print("fetch_mldata")
mnist = fetch_mldata('MNIST original', data_home="data")

# 
num_data=-1
n_steps=30
min_steps=20
max_steps=50
random_steps=True

target_labels=list(range(0,10))
rot_velocity=10.0
flip_noise_rate=0.2
missing_rate=0.1
save_missing_mask=True
generate_interval=True
interval_step=2
generate_input=True
max_steps=50
base_path=""
save_path="rotMNIST/data"
save_datafile="rotMNIST/rotMNIST.json"
save_labelfile="rotMNIST/rotMNIST.label.json"
plot_num=10
plot_steps=20
save_plt="rotMNIST/rotMNIST.png"
seed(1234)
np.random.seed(seed=1234)
#
def add_flip_noise(x,noise_rate=0.2):
	noise=np.random.random_sample(x.shape)
	noise[noise>noise_rate]=1
	noise[noise<=noise_rate]=0
	y=x*noise+(1-x)*(1-noise)
	mask=noise
	return y,mask

def add_zero_noise(x,noise_rate=0.2):
	noise=np.random.random_sample(x.shape)
	noise[noise>noise_rate]=1
	noise[noise<=noise_rate]=0
	y=x*noise
	mask=noise
	return y,mask

def add_square(x,size=10,length=3):
	pos=np.random.randint(x.shape[0])
	x[pos:pos+length,0:size,0:size]=1
	return x

def save_npy(save_file,obj):
	np.save(save_file,obj)
	print("[SAVE]",save_file)
	return save_file

def build_sequence(img,target_label):
	return_data={}
	# 画像サイズの取得(横, 縦)
	size = tuple([img.shape[1], img.shape[0]])
	center = tuple([int(size[0]/2), int(size[1]/2)])
	scale = 1.0
	offset_angle=np.random.normal()*30
	#offset_angle=rand()*360
	
	# determination of verocity
	if rand()>0.5:
		# L
		velocity=rot_velocity
		label=str(target_label)+"L"
	else:
		# R
		velocity=-1*rot_velocity
		label=str(target_label)+"R"
	# generating steps
	steps=n_steps
	if random_steps:
		steps=np.random.poisson(n_steps)
		if steps>max_steps:
			steps=max_steps
		if steps<min_steps:
			steps=min_steps
	# generating interval
	interval=None
	if generate_interval:
		interval=1+np.random.exponential(interval_step-1,size=steps).astype(np.int)
		return_data["interval"]=interval
	# generating input (velocity)
	if generate_input:
		r=np.random.random_sample((steps,))
		input_mean=0
		input_rate=0.2
		r[r>=(1-input_rate)]=1
		r[r<(1-input_rate)]=0
		#input_seq=np.random.exponential(input_mean,size=steps)
		input_seq=r*np.random.normal(input_mean,size=steps)
		return_data["input"]=input_seq
	seq=[]
	angle_seq=[]
	angle=offset_angle
	for j in range(steps):
		# rotation
		rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
		img_rot = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
		seq.append(img_rot)
		angle_seq.append(angle)
		angle +=(velocity+input_seq[i])*interval[i]
	seq=np.array(seq)/255.0
	angle_seq=np.array(angle_seq)
	
	# adding noises
	seq=add_square(seq)
	seq,_=add_flip_noise(seq,flip_noise_rate)
	seq,mask=add_zero_noise(seq,missing_rate)
	seq=seq.reshape(steps,28*28)
	mask=mask.reshape(steps,28*28)
	return_data["data"]=seq
	return_data["state"]=angle_seq
	if save_missing_mask:
		return_data["mask"]=mask
	return return_data,label



os.makedirs(save_path,exist_ok=True)
#p = np.random.random_integers(0, len(mnist.data), 1)

mnist_data = mnist.data
mnist_target = mnist.target

mnist_data = mnist_data[mnist_target < 5]
mnist_target = mnist_target[mnist_target < 5]

idx=list(range(len(mnist_data)))
np.random.shuffle(idx)
idx = idx[:3001]

if num_data>0:
	idx = idx[0:num_data]
#
# rotMNIST.label.json
rot_label_data={}
rot_label_mapping={}
n_label=len(target_labels)
for i in range(n_label):
	rot_label_data[i]=str(i)+"L"
	rot_label_data[i+n_label]=str(i)+"R"
	rot_label_mapping[rot_label_data[i]]=i
	rot_label_mapping[rot_label_data[i+n_label]]=i+n_label
fp=open(save_labelfile,"w")
json.dump(rot_label_data,fp,indent=4)
print("[SAVE]",save_labelfile)


#
mnist_arr=np.array(list(zip(mnist_data, mnist_target)))
rot_data={}
plot_data=[]
plot_label=[]
for index, val in enumerate(mnist_arr[idx]):
	data=val[0]
	target_label=int(val[1])
	if target_label not in target_labels:
		continue
	img=data.reshape(28,28)
	sequence,label=build_sequence(img,target_label)
	
	# storing sequences for plot
	if len(plot_data)<plot_num:
		plot_data.append(sequence["data"][0:plot_steps])
		plot_label.append(label)
	# saving and constructing dataset
	key=save_path+"/"+"%06d"%(index)
	rot_data[key]={}
	for seq_name,val in sequence.items():
		filename=save_npy(save_path+"/"+"%06d"%(index)+"."+seq_name+".npy",val)
		rot_data[key][seq_name]=[base_path+filename]
	rot_data[key]["label"]=rot_label_mapping[label]
fp=open(save_datafile,"w")
json.dump(rot_data,fp,indent=4)
print("[SAVE]",save_datafile)

cnt=1
for index in range(plot_num):
	data=plot_data[index]
	for j in range(plot_steps):
		plt.subplot(plot_num, plot_steps,cnt)
		plt.axis('off')
		plt.imshow(data[j].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest')
		cnt+=1
#plt.title('%i' % label)
#plt.legend(loc='lower right')
plt.savefig(save_plt)
print("[SAVE]",save_plt)
plt.clf()


