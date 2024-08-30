import errno
import hashlib
import os
import shutil
import tarfile
import time
import urllib
import warnings
from zipfile import ZipFile

import numpy as np
import torch
import torchaudio
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms

import soundfile as sf

import ai8x

class AudioAutoencoder:

	url_fanaudio = "https://zenodo.org/records/3384388/files/-6_dB_fan.zip?download=1"
	fs = 16000

	class_dict = {'normal': 0, '_unknown_': 1}
	dataset_dict = {'OneClass': ('normal', '_unknown_')}


	def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None, augmentation=None, download=False, save_unquantized=False):

		self.root = root
		self.classes = classes
		self.d_type = d_type
		self.t_type = t_type
		self.transform = transform
		self.save_unquantized = save_unquantized

		# self.__parse_quantization(quantization_scheme)
		# self.__parse_augmentation(augmentation)

		if not self.save_unquantized:
			self.data_file = 'dataset.pt'
		else:
			self.data_file = 'unquantized.pt'

		if download:
			self.__download()

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
            ai8x.normalize(args=args)
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