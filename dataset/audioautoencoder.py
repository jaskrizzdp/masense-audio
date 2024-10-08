
import errno
import hashlib
import os
import sys
import shutil
import tarfile
import time
import urllib
import warnings
import glob
import pickle
import librosa
from zipfile import ZipFile

import numpy as np
import torch
import torchaudio
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms

import soundfile as sf

import ai8x

param = {
	"feature" : {
		"n_mels": 64,
		"frames" : 5,
		"n_fft": 1024,
		"hop_length": 512,
		"power": 2.0,
	}
}

def block_average(data, factor):
	num_blocks = len(data) // factor
	data_reshaped = data[:num_blocks * factor].reshape(-1, factor)
	return data_reshaped.mean(axis=1).astype(np.float32)

class AudioAutoencoder:

	url_fanaudio = "https://zenodo.org/records/3384388/files/-6_dB_fan.zip?download=1"
	fs = 16000

	class_dict = {'normal': 0, '_unknown_': 1}
	dataset_dict = {'OneClass': ('normal', '_unknown_')}

	TRAIN = np.uint(0)
	TEST = np.uint(1)
	VALIDATION = np.uint(2)

	def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None, augmentation=None, download=False, save_unquantized=False, target_size=(128, 128)):

		self.root = root
		self.classes = classes
		self.d_type = d_type
		self.t_type = t_type
		self.transform = transform
		self.save_unquantized = save_unquantized

		self.target_size = target_size

		self.__parse_quantization(quantization_scheme)
		self.__parse_augmentation(augmentation)

		if not self.save_unquantized:
			self.data_file = 'dataset.pt'
		else:
			self.data_file = 'unquantized.pt'

		# if download:
		# 	self.__download()

		self.__gen_datasets()

		# self.data, self.targets, self.data_type, self.shift_limits = \
		# 	torch.load(os.path.join(self.processed_folder, self.data_file))

	def __download(self):
		if self.__check_exists():
			return
		
		self.__makedir_exist_ok(self.raw_folder)
		self.__makedir_exist_ok(self.processed_folder)

		filename = "-6_dB_fan.zip"
		self.__download_and_extract_archive(self.url_fanaudio,
											download_root=self.raw_folder,
											filename=filename)


	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')

	def __parse_quantization(self, quantization_scheme):
		if quantization_scheme:
			self.quantization = quantization_scheme
			if 'bits' not in self.quantization:
				self.quantization['bits'] = 8
			if self.quantization['bits'] == 0:
				self.save_unquantized = True
			if 'compand' not in self.quantization:
				self.quantization['compand'] = False
			if 'mu' not in self.quantization:
				self.quantization['mu'] = 255  # Default, ignored when 'compand' is False
		else:
			print('Undefined quantization schema! ',
					'Number of bits set to 8.')
			self.quantization = {'bits': 8, 'compand': False, 'mu': 255}

	def __parse_augmentation(self, augmentation):
		self.augmentation = augmentation
		if augmentation:
			if 'aug_num' not in augmentation:
				print('No key `aug_num` in input augmentation dictionary! ',
						'Using 0.')
				self.augmentation['aug_num'] = 0
			elif self.augmentation['aug_num'] != 0:
				if 'snr' not in augmentation:
					print('No key `snr` in input augmentation dictionary! ',
							'Using defaults: [Min: -5.0, Max: 20.0]')
					self.augmentation['snr'] = {'min': -5.0, 'max': 20.0}
				if 'shift' not in augmentation:
					print('No key `shift` in input augmentation dictionary! '
							'Using defaults: [Min:-0.1, Max: 0.1]')
					self.augmentation['shift'] = {'min': -0.1, 'max': 0.1}

	def __makedir_exist_ok(self, dirpath):
		try:
			os.makedirs(dirpath)
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise

	def __check_exists(self):
		return os.path.exists(os.path.join(self.processed_folder, self.data_file))

	def __gen_bar_updater(self):
		pbar = tqdm(total=None)

		def bar_update(count, block_size, total_size):
			if pbar.total is None and total_size:
				pbar.total = total_size
			progress_bytes = count * block_size
			pbar.update(progress_bytes - pbar.n)

		return bar_update

	def __download_and_extract_archive(self, url, download_root, extract_root=None, filename=None,
										md5=None, remove_finished=False):
		download_root = os.path.expanduser(download_root)
		if extract_root is None:
			extract_root = download_root
		if not filename:
			filename = os.path.basename(url)

		self.__download_url(url, download_root, filename, md5)

		archive = os.path.join(download_root, filename)
		print(f"Extracting {archive} to {extract_root}")
		self.__extract_archive(archive, extract_root, remove_finished)

	def __download_url(self, url, root, filename=None, md5=None):
		root = os.path.expanduser(root)
		if not filename:
			filename = os.path.basename(url)
		fpath = os.path.join(root, filename)

		self.__makedir_exist_ok(root)

		# downloads file
		if self.__check_integrity(fpath, md5):
			print('Using downloaded and verified file: ' + fpath)
		else:
			try:
				print('Downloading ' + url + ' to ' + fpath)
				urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
			except (urllib.error.URLError, IOError) as e:
				if url[:5] == 'https':
					url = url.replace('https:', 'http:')
					print('Failed download. Trying https -> http instead.'
							' Downloading ' + url + ' to ' + fpath)
					urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
				else:
					raise e

	def __extract_archive(self, from_path,
							to_path=None, remove_finished=False):
		if to_path is None:
			to_path = os.path.dirname(from_path)

		if from_path.endswith('.tar.gz'):
			with tarfile.open(from_path, 'r:gz') as tar:
				tar.extractall(path=to_path, filter='data')
		elif from_path.endswith('.zip'):
			with ZipFile(from_path) as archive:
				archive.extractall(to_path)
		else:
			raise ValueError(f"Extraction of {from_path} not supported")

		if remove_finished:
			os.remove(from_path)

	def __calculate_md5(self, fpath, chunk_size=1024 * 1024):
		md5 = hashlib.md5()
		with open(fpath, 'rb') as f:
			for chunk in iter(lambda: f.read(chunk_size), b''):
				md5.update(chunk)
		return md5.hexdigest()

	def __check_md5(self, fpath, md5, **kwargs):
		return md5 == self.__calculate_md5(fpath, **kwargs)

	def __check_integrity(self, fpath, md5=None):
		if not os.path.isfile(fpath):
			return False
		if md5 is None:
			return True
		return self.__check_md5(fpath, md5)
	
	def __reshape_audio(self, audio, row_len=(256)):
		if not isinstance(audio, torch.Tensor):
			audio = torch.tensor(audio, dtype=torch.float32)
		
		audio = audio.numpy()
		audio = block_average(audio, 16)

		num_channels = row_len
		signal_len = 3
		required_size = num_channels * signal_len
		if len(audio) < required_size:
			audio = np.pad(audio, (0, required_size - len(audio)), mode='constant')
		else:
			audio = audio[:required_size]

		# total_size = target_shape[0] * target_shape[1]
		# audio = audio[:total_size]
		
		audio = audio.reshape((num_channels, signal_len))
		# audio = audio.T

		audio = torch.tensor(audio, dtype=torch.float32)
		return audio
	
	def __gen_datasets(self):
		# Load base_directory list
		dirs = sorted(glob.glob(os.path.join(self.raw_folder, "fan", "id_00")))

		# Loop over the base directory
		for dir_idx, target_dir in enumerate(dirs):
			print("\n===========================")
			print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

			# Dataset parameters
			db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
			machine_type = os.path.split(os.path.split(target_dir)[0])[1]
			machine_id = os.path.split(target_dir)[1]

			train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(
				pickle=self.processed_folder,
				machine_type=machine_type,
				machine_id=machine_id,
				db=db
			)
			train_labels_pickle = "{pickle}/train_labels_{machine_type}_{machine_id}_{db}.pickle".format(
				pickle=self.processed_folder,
				machine_type=machine_type,
				machine_id=machine_id,
				db=db
			)
			eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
				pickle=self.processed_folder,
				machine_type=machine_type,
				machine_id=machine_id,
				db=db
			)
			eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
				pickle=self.processed_folder,
				machine_type=machine_type,
				machine_id=machine_id,
				db=db
			)

			# dataset generator
			print("============== DATASET_GENERATOR ==============")
			if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
				if self.d_type == "train":
					self.data = load_pickle(train_pickle)
					self.labels = load_pickle(train_labels_pickle)
				else:
					self.data = load_pickle(eval_files_pickle)
					self.labels = load_pickle(eval_labels_pickle)
				print(type(self.data))
			else:
				train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
				train_data = list_to_vector_array(train_files, target_size=(256, 3))

				if self.d_type == "train":
					self.data = train_data
					self.labels = train_labels

					save_pickle(train_pickle, train_data)
					save_pickle(train_labels_pickle, train_labels)
				else:

					eval_data = []
					for num, file_path in tqdm(enumerate(eval_files), total=len(eval_files)):
						try:
							data = file_to_vector_array(file_path, target_size=(256, 3))
							eval_data.append(data)
						except:
							print("File broken!!: {}".format(file_path))

					# print("eval_files: ", eval_files)

					# eval_data = list_to_vector_array(eval_data,
					# 				msg="generate train_dataset",
					# 				n_mels=param["feature"]["n_mels"],
					# 				frames=param["feature"]["frames"],
					# 				n_fft=param["feature"]["n_fft"],
					# 				hop_length=param["feature"]["hop_length"],
					# 				power=param["feature"]["power"])

					save_pickle(eval_files_pickle, eval_data)
					save_pickle(eval_labels_pickle, eval_labels)

					self.data = eval_data
					self.labels = eval_labels
	
	def __len__(self):
			if self.d_type == "train":
				return self.data.shape[1]
			else:
				return len(self.data)

	def __getitem__(self, index):
	
		inp = self.data[index]
		target = self.labels[index]

        # reshape to 2D
		inp = self.__reshape_audio(inp)
		
		if not self.save_unquantized:
			inp /= 256

		inp = inp.numpy().astype('float32')

		# if self.transform is not None:
		# 	inp = self.transform(inp)
		inp = torch.tensor(inp, dtype=torch.float32)
		target = torch.tensor(target, dtype=torch.float32)

		return inp, target

def dataset_generator(target_dir, normal_dir_name="normal", abnormal_dir_name="abnormal", ext='wav'):
	print("target_dir : {}".format(target_dir))
		
	normal_files = sorted(glob.glob(os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir, normal_dir_name=normal_dir_name, ext=ext))))
	normal_labels = np.zeros(len(normal_files))
	if len(normal_files) == 0:
		print("no_wav_data!!")

	abnormal_files = sorted(glob.glob(os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir, abnormal_dir_name=abnormal_dir_name, ext=ext))))
	abnormal_labels = np.ones(len(abnormal_files))
	if len(abnormal_files) == 0:
		print("no_wav_data!!")

		
	train_files = normal_files[len(abnormal_files):]
	train_labels = normal_labels[len(abnormal_files):]
	eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
	eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
	print("train_file num : {num}".format(num=len(train_files)))
	print("eval_file  num : {num}".format(num=len(eval_files)))

	return train_files, train_labels, eval_files, eval_labels

def file_load(wav_name, mono=False):
	try:
		return librosa.load(wav_name, sr=None, mono=mono)
	except:
		print("file_broken or not exists!! : {}".format(wav_name))

def demux_wav(wav_name, channel=0):
	try:
		multi_channel_data, sr = file_load(wav_name)
		if multi_channel_data.ndim <= 1:
			return sr, multi_channel_data
			
		return sr, np.array(multi_channel_data)[channel, :]
		
	except ValueError as msg:
		print(f'{msg}')

def file_to_vector_array(file_path, target_size=(256, 3)):
	audio, sr = librosa.load(file_path, sr=None)

	num_samples = target_size[0] * target_size[1]
	if len(audio) < num_samples:
		audio = np.pad(audio, (0, num_samples - len(audio)), mode='constant')
	else:
		audio = audio[:num_samples]

	audio = audio.reshape(target_size).astype(np.float32)
	return audio

def list_to_vector_array(file_list, msg="calc...", target_size=(256, 3)):
	vectors = [file_to_vector_array(file, target_size) for file in file_list]
	return np.array(vectors, dtype=np.float32)
	
def save_pickle(filename, save_data):
	print("save_pickle -> {}".format(filename))
	with open(filename, 'wb') as sf:
		pickle.dump(save_data, sf)

def load_pickle(filename):
	print("load_pickle <- {}".format(filename))
	with open(filename, 'rb') as lf:
		load_data = pickle.load(lf)
	return load_data
	
def get_datasets(data, load_train=True, load_test=True, dataset_name='OneClass', quantized=True):
	(data_dir, args) = data

	if quantized:
		transform = transforms.Compose([
		transforms.ToTensor(),
			ai8x.normalize(args=args),
		])
	else:
		transform = None

	classes = AudioAutoencoder.dataset_dict['OneClass']

	if dataset_name != 'OneClass':
		raise ValueError(f'Invalid dataset name {dataset_name}. Expected "OneClass".')
	
	if quantized:
		augmentation = {'aug_num': 2, 'shift': {'min': -0.1, 'max': 0.1}, 'snr': {'min': -5.0, 'max': 20.}}
		quantization_scheme = {'compand': False, 'mu': 10}
	else:
		augmentation = {'aug_num': 0, 'shift': {'min': -0.1, 'max': 0.1}, 'snr': {'min': -5.0, 'max': 20.}}
		quantization_scheme = {'bits': 0}

	if load_train:
		train_dataset = AudioAutoencoder(root=data_dir, classes=classes, d_type='train', transform=transform, t_type='one_class', quantization_scheme=quantization_scheme, augmentation=augmentation, download=True)
	else:
		train_dataset = None

	if load_test:
		test_dataset = AudioAutoencoder(root=data_dir, classes=classes, d_type='test', transform=transform, t_type='one_class', quantization_scheme=quantization_scheme, augmentation=augmentation, download=True)

		if args.truncate_testset:
			test_dataset.data = test_dataset.data[:1]

	else:
		test_dataset = None

	return train_dataset, test_dataset
	
datasets = [
	{
		'name': 'AudioAutoencoder',
		'input': (256, 3),
		'output': AudioAutoencoder.dataset_dict['OneClass'],
		'weight': (1, 1),
		'loader': get_datasets,
	}
]