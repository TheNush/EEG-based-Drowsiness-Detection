import  sys
import numpy as np
import pyedflib
from sklearn.preprocessing import scale
import random

random.seed(25)
edf_file = sys.argv[1]
annotations_file = sys.argv[2]
sleep_stages_dict = {'Sleep stage W':5, 'Sleep stage 1':3, 'Sleep stage 2':2, 'Sleep stage 3':1,'Sleep stage 4':0, 'Sleep stage R':4, 'Movement time':6}

def read_annotations(annotations_file, sleep_stages_dict):
	print('Reading Annotations File...')
	with open(annotations_file) as x:
		content = x.readlines()
	nr_states = len(sleep_stages_dict)
	sleep_stages = []
	for index in range(1, len(content)-1):
		tokens = content[index].split(',')
		sleep_stages = sleep_stages + [sleep_stages_dict[tokens[len(tokens)-1][:-1]]]
	return sleep_stages


sleep_stages = read_annotations(annotations_file, sleep_stages_dict)
sleep_stages = np.array(sleep_stages)
nr_states = len(np.unique(sleep_stages))
real_sleep_stages = sleep_stages[1:-1]
sleep_stages = sleep_stages.reshape((len(sleep_stages),1))

def prep_epochs(edf_file, epoch_len, sf):
	print('Preparing Epochs...')
	f = pyedflib.EdfReader(edf_file)
	n = f.signals_in_file
	labels = f.getSignalLabels()
	sig = f.readSignal(0)
	L = epoch_len * sf
	epochs = np.reshape(sig, (-1, L))
	return epochs

epochs = prep_epochs(edf_file, epoch_len=30, sf=100)

def extract_features(epochs, epoch_len, sf):
	print('Extracting Features...')
	n_epochs = epochs.shape[0]
	epoch_sam = epoch_len * sf
	f = np.linspace(0, epoch_sam-1, epoch_sam)/epoch_len
	
	delta1, delta2, theta1, theta2, alpha1, alpha2, beta1, beta2 = 0,4,4,8,8,13,13,30

	all_indices = np.where((f<=beta2))
	delta_indices = np.where((f >= delta1) & (f <= delta2))
	theta_indices = np.where((f >= theta1) & (f <= theta2))
	alpha_indices = np.where((f >= alpha1) & (f <= alpha2))
	beta_indices = np.where((f >= beta1) & (f <= beta2))
	n_features = 6
	features = np.zeros((n_epochs, n_features))
	for index in range(n_epochs):
		epoch = epochs[index][:]
		fft = abs(np.fft.fft(epoch))
		mean_tot_power = np.mean(fft[all_indices])
		features[index, :] = (mean_tot_power, np.mean(f[all_indices] * fft[all_indices])/mean_tot_power, 
			np.mean(f[delta_indices] * fft[delta_indices])/mean_tot_power, np.mean(f[theta_indices] * fft[theta_indices])/mean_tot_power,
			np.mean(f[alpha_indices] * fft[alpha_indices])/mean_tot_power, np.mean(f[beta_indices] * fft[beta_indices])/mean_tot_power)
	return scale(features), mean_tot_power

features,mean_power = extract_features(epochs, epoch_len=30, sf=100)

nr_groups = 20

def ranges(start, end, nb):
	step = (end-start)/nb
	return [((start + (step*i)), start+(step*(i+1))) for i in range(nb)]

def codebook(features, nr_groups):
	print('Passing through Codebook...')
	n_epochs = features.shape[0]
	n_feats = features.shape[1]
	mins = np.min(features, 0)
	maxs = np.max(features, 0)
	min_max = list(zip(mins, maxs))
	intervals = []
	
	for index in range(n_feats):
		intervals.append(ranges(min_max[index][0], min_max[index][1], nr_groups))
	
	book = np.array([[random.uniform(min_max[column][0], min_max[column][1])for column in range(n_feats)]for row in range(nr_groups)])
	epoch_codes = np.zeros(n_epochs, dtype=np.int)
	epoch_codes_prev = np.zeros(n_epochs, dtype=np.int)

	while True:
		for epoch_index in n_epochs:
			distances = np.zeros(nr_groups)
			for book_index in range(nr_groups):
				distances[book_index] = 
	return book

res = codebook(features, nr_groups)
print(res.shape)