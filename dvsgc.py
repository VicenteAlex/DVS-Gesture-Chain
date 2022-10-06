from torchvision.datasets import DatasetFolder
from torchvision.datasets import utils

from typing import Callable, Dict, Optional, Tuple
import numpy as np
import spikingjelly.datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import random
random.seed(123)
from spikingjelly.configure import max_threads_number_for_datasets_preprocess
from tqdm import tqdm


def make_class_combinations(seq_len, class_num, repeat):
    """
    seq_len: Sequence length
    class_num: number of classes to use
    repeat: whether to allow repetition of the same gesture in subsequent positions
    """
    permuted_classes = ['1','3','8','7','9','0','2','4','6','5']
    permuted_classes = permuted_classes[:class_num]
    n_combs = class_num ** seq_len
    new_class_list = [''] * n_combs

    def fill_level(new_class_list):
        cl = 0
        cl_count = 0
        prev_combs = len(new_class_list[0])
        for n in range(n_combs):
            new_class_list[n] = new_class_list[n] + permuted_classes[cl]
            cl_count +=1
            if cl_count == class_num**prev_combs:  #int(n_combs/class_num):
                cl += 1
                if cl > class_num-1:
                    cl = 0
                cl_count = 0
        return new_class_list

    for lvl in range(seq_len):
        new_class_list = fill_level(new_class_list)

    if not repeat:
        corrected_class_list = new_class_list.copy()
        for nc in range(len(new_class_list)):
            prev_num = ''
            for num in new_class_list[nc]:
                if num == prev_num:
                    corrected_class_list.remove(new_class_list[nc])
                    break
                prev_num = num
        new_class_list = corrected_class_list

    return new_class_list


def create_combined_directory_structure(source_dir: str, target_dir: str, class_list, validation=False) -> None:
    """
    :param source_dir: Path of the directory that be copied from
    :type source_dir: str
    :param target_dir: Path of the directory that be copied to
    :type target_dir: str
    :param class_list: List of classes to create
    :param validation: Percentage of train instances to use for validation
    :return: None

    Create the same directory structure in ``target_dir`` with that of ``source_dir``.
    """
    for sub_dir_name in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir_name)
        if os.path.isdir(source_sub_dir):
            target_sub_dir = os.path.join(target_dir, sub_dir_name)
            os.mkdir(target_sub_dir)
            print(f'Mkdir [{target_sub_dir}].')

            for cl in class_list:
                cl_dir = os.path.join(target_sub_dir,cl)
                if not os.path.exists(cl_dir):
                    os.mkdir(cl_dir)
    if validation:
        target_sub_dir = os.path.join(target_dir, 'validation')
        os.mkdir(target_sub_dir)
        print(f'Mkdir [{target_sub_dir}].')

        for cl in class_list:
            cl_dir = os.path.join(target_sub_dir, cl)
            if not os.path.exists(cl_dir):
                os.mkdir(cl_dir)


def write_combined_events(combined_cl, source_events_np_root, frames_np_root, frames_number, split_by, H, W,
                          alpha_min, alpha_max, validation=None):

    if validation is not None:
        sets = ['train', 'test', 'validation']
        origin_list = []
        for c in os.listdir(os.path.join(source_events_np_root, 'train')):
            for e in os.listdir(os.path.join(source_events_np_root, 'train', c)):
                if e not in origin_list:
                    origin_list.append(e)
        origin_list.sort()
        train_list = origin_list[:int(len(origin_list) * (1 - validation))]
        validation_list = origin_list[int(len(origin_list) * (1 - validation)):]
    else:
        sets = ['train', 'test']

    for set in sets:
        event_root_list = []
        n_combs = len(combined_cl)
        for i in range(n_combs):
            if set == 'validation':
                event_root_list.append(
                    os.path.join(source_events_np_root, 'train', combined_cl[i]))  # list of the necessary classes
            else:
                event_root_list.append(os.path.join(source_events_np_root, set, combined_cl[i]))  # list of the necessary classes

        cl_event_files = os.listdir(event_root_list[0])  # filenames in the first gesture folder to combine

        if validation is not None and set != 'test':
            # take out from cl_event files
            original_cl_event_files = cl_event_files.copy()
            for i in range(len(original_cl_event_files)):

                if set == 'train':
                    ref_list = train_list
                if set == 'validation':
                    ref_list = validation_list

                if original_cl_event_files[i] not in ref_list:
                    cl_event_files.remove(original_cl_event_files[i])

        for e_fn in cl_event_files:

            if all([os.path.exists(os.path.join(e_root, e_fn)) for e_root in event_root_list]):

                output_dir = os.path.join(frames_np_root, set, combined_cl)
                # with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(),
                #                                        max_threads_number_for_datasets_preprocess)) as tpe:

                gesture_frames = int(frames_number / (n_combs * 0.6))  # Total frames of a single gesture
                min_len = gesture_frames * alpha_min
                max_len = gesture_frames * alpha_max

                previous_len = 0

                frames_list = []
                for i in range(n_combs):
                    if i == n_combs - 1:  # force last to complete up to frames number
                        current_len = frames_number - previous_len
                    else:
                        low_limit = frames_number - previous_len - (n_combs - 1 - i) * max_len
                        high_limit = frames_number - previous_len - (n_combs - 1 - i) * min_len
                        current_len = int(random.uniform(max(min_len, low_limit), min(max_len, high_limit)))
                    current_frames = sjds.integrate_events_by_fixed_frames_number(
                        np.load(os.path.join(event_root_list[i], e_fn)), split_by, gesture_frames, H, W) # allow_pickle=True
                    frames_list.append(current_frames[:current_len, :, :, :])
                    previous_len += current_len

                frames = np.concatenate(frames_list, axis=0)

                fname = os.path.join(output_dir, e_fn)
                np.savez_compressed(fname, frames=frames)


class DVSGestureChain(DatasetFolder):
    def __init__(
            self,
            root: str,
            frames_number: int,
            split: str,
            validation: float = 0.2,
            split_by: str = 'number',
            alpha_min: float = 0.5,
            alpha_max: float = 0.7,
            seq_len: int = 4,
            class_num: int = 3,
            repeat: bool = True,
            dvsg_path: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            root: root path of the dataset
            frames_number: the integrated frame number
            split: split from: ['train', 'validation', 'test']
            validation: fraction of the training set to use for validation
            data_type: `event` or `frame`
            split_by: `time` or `number`
            alpha_min: lower bound for the gesture duration as a factor of its total duration
            alpha_max: upper bound for the gesture duration as a factor of its total duration
            seq_len: number of gestures in the chain
            class_num: number of classes to use (up to 11)
            repeat: whether to allow repetition of the same gesture twice in a row
            dvsg_path: If DVS-Gesture events are saved in another folder they can be retrieved by indicating the path
             here, else they will be created again in root
            transform: a function/transform that takes in a sample and returns a transformed version.
            target_transform: a function/transform that takes in the target and transforms it.
        """
        assert split in ['train','validation','test']

        if dvsg_path is not None and os.path.exists(dvsg_path):
            source_events_np_root = dvsg_path
        else:
            source_events_np_root = os.path.join(root, 'events_np')

        events_np_root = os.path.join(root, 'events_np')

        frames_np_root = os.path.join(root, f'DVSGC_frames_number_{frames_number}_split_by_{split_by}')

        if not os.path.exists(source_events_np_root) and not os.path.exists(frames_np_root):

            download_root = os.path.join(root, 'download')

            if os.path.exists(download_root):
                print(f'The [{download_root}] directory for saving downloaded files already exists, check files...')
                # check files
                resource_list = self.resource_url_md5()
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')

                        if os.path.exists(fpath):
                            # If file is corrupted, we will remove it.
                            os.remove(fpath)
                            print(f'Remove [{fpath}]')

                        if self.downloadable():
                            # If file does not exist, we will download it.
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            raise NotImplementedError(
                                f'This dataset can not be downloaded automatically, please download [{file_name}] from [{url}] manually and put files at {download_root}.')

            else:
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                resource_list = self.resource_url_md5()
                if self.downloadable():
                    # download and extract file
                    for i in range(resource_list.__len__()):
                        file_name, url, md5 = resource_list[i]
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(f'This dataset can not be downloaded automatically, '
                                              f'please download files manually and put files at [{download_root}]. '
                                              f'The resources file_name, url, and md5 are: \n{resource_list}')

            # We have downloaded files and checked files. Now, let us extract the files
            extract_root = os.path.join(root, 'extract')
            if os.path.exists(extract_root):
                print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                      f'The data integrity of the extracted files will not be checked.\n'
                      f'If the extracted files are not integrated, please delete [{extract_root}] manually, '
                      f'then the files will be re-extracted from [{download_root}].')
                # shutil.rmtree(extract_root)
                # print(f'Delete [{extract_root}].')
            else:
                os.mkdir(extract_root)
                print(f'Mkdir [{extract_root}].')
                self.extract_downloaded_files(download_root, extract_root)

            # Now let us convert the origin binary files to npz files
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
            print(f'Start to convert the origin data from [{extract_root}] to [{events_np_root}] in np.ndarray format.')
            self.create_events_np_files(extract_root, events_np_root)

        H, W = self.get_H_W()

        # Create DVS-GC frames from DVS-Gesture events:

        if os.path.exists(frames_np_root):
            print(f'The directory [{frames_np_root}] already exists.')
            print(f'Checking integrity')

            set_fold = split

            for class_fold in os.listdir(os.path.join(frames_np_root,set_fold)):
                if validation is not None:
                    train_size = int(97 * (1-validation))
                else:
                    train_size = 97
                if (set_fold == 'train' and len(os.listdir(os.path.join(frames_np_root,set_fold,class_fold))) < train_size)\
                        or (set_fold == 'test' and len(os.listdir(os.path.join(frames_np_root,set_fold,class_fold))) < 24):
                    print(f'Complete [{os.path.join(set_fold,class_fold)}]')
                    write_combined_events(class_fold, source_events_np_root, frames_np_root, frames_number, split_by,
                                          H, W, alpha_min=alpha_min, alpha_max=alpha_max, validation=validation)
            print(f'Done')

        else:
            os.mkdir(frames_np_root)
            print(f'Mkdir [{frames_np_root}].')
            # create the same directory structure
            class_list = make_class_combinations(seq_len=seq_len, class_num=class_num, repeat=repeat)
            val_bool = validation is not None
            create_combined_directory_structure(source_events_np_root, frames_np_root, class_list,
                                                validation=val_bool)
            t_ckp = time.time()

            for combined_cl in tqdm(class_list):
                write_combined_events(combined_cl, source_events_np_root, frames_np_root, frames_number, split_by,
                                      H, W, alpha_min=alpha_min, alpha_max=alpha_max, validation=validation)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

        _root = frames_np_root
        _loader = sjds.load_npz_frames
        _transform = transform
        _target_transform = target_transform

        _root = os.path.join(_root, split)

        super().__init__(root=_root, loader=_loader, extensions='.npz', transform=_transform,
                         target_transform=_target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        return sjds.load_aedat_v3(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVSGestureChain.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np.savez_compressed(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1


    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe:
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVSGestureChain.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir)

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        tpe.submit(DVSGestureChain.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(aedat_dir, fname + '_labels.csv'), test_dir)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128