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
from datasets.msnoise import MSnoise
from datasets.signalmixer import SignalMixer


class AudioAutoencoder:

    # all dataset = "https://zenodo.org/records/3384388"
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

    if download:
        self.__download()

    self.data, self.targets, self.data_type, self.shift_limits = \
        torch.load(os.path.join(self.processed_folder, self.data_file))

<<<<<<< Updated upstream
    print(f'\nProcessing {self.d_type}...')
    self.__filter_dtype()

    self.__filter_classes()
=======
# HEAD
		filename = "fanaudio"
		self.__download_and_extract_archive(self.url_fanaudio,
											download_root=self.raw_folder,
											filename=filename)
		filename = "-6_dB_fan.zip"
		self.__download_and_extract_archive(self.url_fanaudio,
						download_root=self.raw_folder,
						filename=filename)
# 7ad98a40ea0c47c2baa6dd97a862d7ba568d0f2b
>>>>>>> Stashed changes

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

def __download(self):
    if self.__check__exists():
        return
    
    self.__makedir_exist_ok(self.raw_folder)
    self.__makedir_exist_ok(self.processed_folder)

    # download fan audio
    filename = self.url_speechcommand.rpartition('/')[2]
    self.__download_and_extract_archive(self.url_fanaudio, download_root=self.raw_folder, filename=filename)

    self.__gen_datasets()

def __sample_wav(self, folder_in, folder_out, sample_num, sr=16000, ext='.wav', exp_len=16384):

        # create output folder
        self.__makedir_exist_ok(folder_out)

        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):
                if filename.endswith(ext):
                    fname = os.path.join(dirpath, filename)
                    data, sr_original = sf.read(fname, dtype='float32')
                    assert sr_original == sr
                    max_start_pt = len(data) - exp_len

                    for i in range(sample_num):
                        start_pt = np.random.randint(0, max_start_pt)
                        audio = data[start_pt:start_pt+exp_len]

                        outfile = os.path.join(folder_out, filename[:-len(ext)] + '_' +
                                               str(f"{i}") + '.wav')
                        sf.write(outfile, audio, sr)
        print('File conversion completed.')

def __gen_bar_updater(self):
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

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

def __resample_convert_wav(self, folder_in, folder_out, sr=16000, ext='.flac'):
        # create output folder
        self.__makedir_exist_ok(folder_out)

        # find total number of files to convert
        total_count = 0
        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):
                if filename.endswith(ext):
                    total_count += 1
        print(f"Total number of speech files to convert to 2-sec .wav: {total_count}")
        converted_count = 0
        # segment each audio file to 1-sec frames and save
        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):

                i = 0
                if filename.endswith(ext):
                    fname = os.path.join(dirpath, filename)
                    data, sr_original = sf.read(fname, dtype='float32')
                    assert sr_original == sr

                    # normalize data
                    mx = np.amax(abs(data))
                    data = data / mx

                    chunk_start = 0
                    frame_count = 0

                    precursor_len = 30 * 128
                    postcursor_len = 196 * 128
                    normal_threshold = 30

                    while True:
                        if chunk_start + postcursor_len > len(data):
                            break

                        chunk = data[chunk_start: chunk_start + 128]
                        # scaled average over 128 samples
                        avg = 1000 * np.average(abs(chunk))
                        i += 128

                        if avg > normal_threshold and chunk_start >= precursor_len:
                            print(f"\r Converting {converted_count + 1}/{total_count} "
                                  f"to {frame_count + 1} segments", end=" ")
                            frame = data[chunk_start - precursor_len:chunk_start + postcursor_len]

                            outfile = os.path.join(folder_out, filename[:-5] + '_' +
                                                   str(f"{frame_count}") + '.wav')
                            sf.write(outfile, frame, sr)

                            chunk_start += postcursor_len
                            frame_count += 1
                        else:
                            chunk_start += 128
                    converted_count += 1
                else:
                    pass
        print(f'\rFile conversion completed: {converted_count} files ')

def __filter_dtype(self):
    if self.d_type == 'train':
        idx_to_select = (self.data_type == self.TRAIN)[:, -1]
    elif self.d_type == 'test':
        idx_to_select = (self.data_type == self.TEST)[:, -1]
    else:
        print(f'Unknown data type: {self.d_type}')
        return

    set_size = idx_to_select.sum()
    print(f'{self.d_type} set: {set_size} elements')

    # Filter the data
    self.data = self.data[idx_to_select, :]
    self.targets = self.targets[idx_to_select, :]
    self.data_type = self.data_type[idx_to_select, :]
    self.shift_limits = self.shift_limits[idx_to_select, :]

    # For training, include validation data if it exists
    if self.d_type == 'train':
        idx_to_select = (self.data_type_original == self.VALIDATION)[:, -1]
        if idx_to_select.sum() > 0:
            self.data = torch.cat((self.data, self.data_original[idx_to_select, :]), dim=0)
            self.targets = torch.cat((self.targets, self.targets_original[idx_to_select, :]), dim=0)
            self.data_type = torch.cat((self.data_type, self.data_type_original[idx_to_select, :]), dim=0)
            self.shift_limits = torch.cat((self.shift_limits, self.shift_limits_original[idx_to_select, :]), dim=0)
            self.valid_indices = range(set_size, set_size + idx_to_select.sum())
            print(f'validation set: {idx_to_select.sum()} elements')

    # Clean up
    del self.data_original
    del self.targets_original
    del self.data_type_original
    del self.shift_limits_original

def __filter_classes(self):
    # Ensure only normal class and possibly an unknown class
    if len(self.class_dict) > 2:
        raise ValueError('More than two classes detected. This method is for one-class classification.')
    
    new_class_label = 0
    # Re-label normal class
    for c in self.classes:
        if c not in self.class_dict:
            if c == '_unknown_':
                continue
            raise ValueError(f'Class {c} not found in data')
        
        if c != '_unknown_':
            num_elems = (self.targets == self.class_dict[c]).cpu().sum()
            print(f'Class {c} (# {self.class_dict[c]}): {num_elems} elements')
            self.targets[(self.targets == self.class_dict[c])] = new_class_label
            new_class_label += 1

    # Handle unknown class if necessary
    num_elems = (self.targets < len(self.class_dict)).cpu().sum()
    print(f'Class _unknown_: {num_elems} elements')
    self.targets[(self.targets < len(self.class_dict))] = new_class_label
    self.targets -= len(self.class_dict)

def __len__(self):
        return len(self.data)

def __reshape_audio(self, audio, row_len=128):
        # add overlap if necessary later on
        return torch.transpose(audio.reshape((-1, row_len)), 1, 0)

@staticmethod
def quantize_audio(data, num_bits=8, compand=False, mu=255):
    """Quantize audio data to a specified bit-depth for AI8x training."""
    
    # Apply companding if required
    if compand:
        data = AudioAutoencoder.compand(data, mu)

    # Compute quantization step size and maximum value based on num_bits
    step_size = 2.0 / (2 ** num_bits)
    max_val = 2 ** num_bits - 1
    
    # Quantize data
    q_data = np.round((data + 1.0) / step_size)
    q_data = np.clip(q_data, 0, max_val)

    # Re-expand if companding was applied
    if compand:
        data_ex = (q_data - 2 ** (num_bits - 1)) / 2 ** (num_bits - 1)
        data_ex = AudioAutoencoder.expand(data_ex)
        q_data = np.round((data_ex + 1.0) / step_size)
        q_data = np.clip(q_data, 0, max_val)

    # Return as uint8 for compatibility with AI8x training requirements
    return np.uint8(q_data)
def speed_augment(self, audio, fs, sample_no=0):
        """Augments audio by randomly changing the speed of the audio.
        The generated coefficient follows 0.9, 1.1, 0.95, 1.05... pattern
        """
        speed_multiplier = 1.0 + 0.2 * (sample_no % 2 - 0.5) / (1 + sample_no // 2)

        sox_effects = [["speed", str(speed_multiplier)], ["rate", str(fs)]]
        aug_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            torch.unsqueeze(torch.from_numpy(audio).float(), dim=0), fs, sox_effects)
        aug_audio = aug_audio.numpy().squeeze()

        return aug_audio, speed_multiplier

def speed_augment_multiple(self, audio, fs, exp_len, n_augment):
    """Calls `speed_augment` function for n_augment times for given audio data.
    Finally the original audio is added to have (n_augment+1) audio data.
    """
    aug_audio = [None] * (n_augment + 1)
    aug_speed = np.ones((n_augment + 1,))
    shift_limits = np.zeros((n_augment + 1, 2))
    voice_begin_idx, voice_end_idx = self.get_audio_endpoints(audio, fs)
    aug_audio[0] = audio
    for i in range(n_augment):
        aug_audio[i+1], aug_speed[i+1] = self.speed_augment(audio, fs, sample_no=i)
    for i in range(n_augment + 1):
        if len(aug_audio[i]) < exp_len:
            aug_audio[i] = np.pad(aug_audio[i], (0, exp_len - len(aug_audio[i])), 'constant')
        aug_begin_idx = voice_begin_idx * aug_speed[i]
        aug_end_idx = voice_end_idx * aug_speed[i]
        if aug_end_idx - aug_begin_idx <= exp_len:
            segment_begin = max(aug_end_idx, exp_len) - exp_len
            segment_end = max(aug_end_idx, exp_len)
            aug_audio[i] = aug_audio[i][segment_begin:segment_end]
            shift_limits[i, 0] = -aug_begin_idx + (max(aug_end_idx, exp_len) - exp_len)
            shift_limits[i, 1] = max(aug_end_idx, exp_len) - aug_end_idx
        else:
            midpoint = (aug_begin_idx + aug_end_idx) // 2
            aug_audio[i] = aug_audio[i][midpoint - exp_len // 2: midpoint + exp_len // 2]
            shift_limits[i, :] = [0, 0]
    return aug_audio, aug_speed, shift_limits


def __gen_datasets(self, exp_len=16384):
    print('Generating dataset from raw data samples for the first time.')
    print('This process may take a few minutes.')
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        # Assuming you have only one class
        labels = [d for d in os.listdir(self.raw_folder) if os.path.isdir(os.path.join(self.raw_folder, d))]
        
        # If you only have one class, adjust label processing
        if len(labels) != 1:
            raise ValueError("Expected exactly one class folder, found: {}".format(len(labels)))

        label = labels[0]
        print(f'Processing the label: {label}')

        record_list = sorted(os.listdir(os.path.join(self.raw_folder, label)))
        record_len = len(record_list)

        # Create arrays to store data
        number_of_total_samples = record_len * (self.augmentation['aug_num'] + 1)
        if not self.save_unquantized:
            data_in = np.empty((number_of_total_samples, exp_len), dtype=np.uint8)
        else:
            data_in = np.empty((number_of_total_samples, exp_len), dtype=np.float32)

        data_shift_limits = np.empty((number_of_total_samples, 2), dtype=np.int16)
        data_class = np.full((number_of_total_samples, 1), 0, dtype=np.uint8)

        time_s = time.time()

        sample_index = 0
        for r, record_name in enumerate(record_list):
            if r % 1000 == 0:
                print(f'\t{r + 1} of {record_len}')

            record_pth = os.path.join(self.raw_folder, label, record_name)
            record, fs = sf.read(record_pth, dtype='float32')

            # Apply speed augmentations and calculate shift limits
            audio_seq_list, shift_limits = \
                self.speed_augment_multiple(record, fs, exp_len, self.augmentation['aug_num'])

            for local_id, audio_seq in enumerate(audio_seq_list):
                if not self.save_unquantized:
                    data_in[sample_index] = \
                        AudioAutoencoder.quantize_audio(audio_seq,
                                           num_bits=self.quantization['bits'],
                                           compand=self.quantization['compand'],
                                           mu=self.quantization['mu'])
                else:
                    data_in[sample_index] = audio_seq
                data_shift_limits[sample_index] = shift_limits[local_id]
                sample_index += 1

        dur = time.time() - time_s
        print(f'Finished in {dur:.3f} seconds.')
        print(data_in.shape)

        data_in = torch.from_numpy(data_in)
        data_class = torch.from_numpy(data_class)
        data_shift_limits = torch.from_numpy(data_shift_limits)

        raw_dataset = (data_in, data_class, data_shift_limits)
        torch.save(raw_dataset, os.path.join(self.processed_folder, self.data_file))

    print('Dataset created.')

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
