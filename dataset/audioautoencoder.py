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

class AudioAutoencoder:

	url_fanaudio = "https://zenodo.org/records/3384388/files/-6_dB_fan.zip?download=1"
	fs = 16000

	class_dict = {'normal': 0, '_unknown_': 1}
	dataset_dict = {'OneClass': ('normal', '_unknown_')}

	TRAIN = np.uint(0)
	TEST = np.uint(1)
	VALIDATION = np.uint(2)

	def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None, augmentation=None, download=False, save_unquantized=False):

		self.root = root
		self.classes = classes
		self.d_type = d_type
		self.t_type = t_type
		self.transform = transform
		self.save_unquantized = save_unquantized

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

	def __gen_datasets(self):

		# load base_directory list
		dirs = sorted(glob.glob(os.path.join(self.raw_folder, "fan", "id_00")))

		# loop of the base directory
		for dir_idx, target_dir in enumerate(dirs):
			print("\n===========================")
			print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

			# dataset param        
			db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
			machine_type = os.path.split(os.path.split(target_dir)[0])[1]
			machine_id = os.path.split(target_dir)[1]

			train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=self.processed_folder,
												machine_type=machine_type,
												machine_id=machine_id, db=db)
			train_labels_pickle = "{pickle}/train_labels_{machine_type}_{machine_id}_{db}.pickle".format(
												pickle=self.processed_folder,
												machine_type=machine_type,
												machine_id=machine_id,
												db=db)
			eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
												pickle=self.processed_folder,
												machine_type=machine_type,
												machine_id=machine_id,
												db=db)
			eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
												pickle=self.processed_folder,
												machine_type=machine_type,
												machine_id=machine_id,
												db=db)
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
				train_data = list_to_vector_array(train_files,
								msg="generate train_dataset",
								n_mels=param["feature"]["n_mels"],
								frames=param["feature"]["frames"],
								n_fft=param["feature"]["n_fft"],
								hop_length=param["feature"]["hop_length"],
								power=param["feature"]["power"])

				if self.d_type == "train":
					self.data = train_data
					self.labels = train_labels

					save_pickle(train_pickle, train_data)
					save_pickle(train_labels_pickle, train_labels)
				else:

					eval_data = []
					for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
						try:
							data = file_to_vector_array(file_name,
										n_mels=param["feature"]["n_mels"],
										frames=param["feature"]["frames"],
										n_fft=param["feature"]["n_fft"],
										hop_length=param["feature"]["hop_length"],
										power=param["feature"]["power"])
							eval_data.append(data)
						except:
							print("File broken!!: {}".format(file_name))

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
		if self.d_type == "train":
			dat = self.data[:, index]
		else:
			dat = self.data[index]
		label = self.labels[index]

		return torch.as_tensor(dat, dtype=torch.float32), torch.tensor(label)

def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : np.array( np.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    print("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = np.zeros(len(normal_files))
    if len(normal_files) == 0:
        print("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = np.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        print("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    print("train_file num : {num}".format(num=len(train_files)))
    print("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels

# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : np.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : np.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, np.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        print(f'{msg}')


# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : np.array( np.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : np.array( np.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    print("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)

def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    print("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data

def get_datasets(data, load_train=True, load_test=True, dataset_name='OneClass',
                 quantized=True):
    """
    Load dataset for one-class audio classification.

    The dataset is loaded from the archive file, so the file is required for this version.
    For one-class classification, we use 'normal' as the positive class and '_unknown_' for unknowns.

    Data is augmented to improve robustness to variations. Specific augmentation settings can be customized.
    """
    (data_dir, args) = data

    if quantized:
        transform = transforms.Compose([
	    transforms.ToTensor(),
            ai8x.normalize(args=args),
        ])
    else:
        transform = None

    # Define classes for one-class classification
    classes = AudioAutoencoder.dataset_dict['OneClass']

    if dataset_name != 'OneClass':
        raise ValueError(f'Invalid dataset name {dataset_name}. Expected "OneClass".')

    # Define augmentation and quantization scheme
    if quantized:
        augmentation = {'aug_num': 2, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'compand': False, 'mu': 10}
    else:
        augmentation = {'aug_num': 0, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'bits': 0}

    # Load datasets
    if load_train:
        train_dataset = AudioAutoencoder(root=data_dir, classes=classes, d_type='train',
                            transform=transform, t_type='one_class',
                            quantization_scheme=quantization_scheme,
                            augmentation=augmentation, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = AudioAutoencoder(root=data_dir, classes=classes, d_type='test',
                           transform=transform, t_type='one_class',
                           quantization_scheme=quantization_scheme,
                           augmentation=augmentation, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
     {
          'name': 'AudioAutoencoder',
          'input': (128, 128),
          'output': AudioAutoencoder.dataset_dict['OneClass'],
          'weight': (1, 1),
          'loader': get_datasets,

     }
]