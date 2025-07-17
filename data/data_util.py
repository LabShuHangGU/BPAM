from os import path as osp
from utils import scandir

def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    print(gt_folder)
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths

def paired_paths_from_folder_ppr10k(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[dict]: Returned path list.
    """
    # assert len(folders) == 2, (
    #     'The len of folders should be 2 with [input_folder, gt_folder]. '
    #     f'But got {len(folders)}')
    # assert len(keys) == 2, (
    #     'The len of keys should be 2 with [input_key, gt_key]. '
    #     f'But got {len(keys)}')
    # input_folder, gt_folder = folders
    # input_key, gt_key = keys

    # # Get the paths from both folders
    # input_paths = list(scandir(input_folder))
    # gt_paths = list(scandir(gt_folder))

    # print(gt_folder)
    
    # # Ensure the length of input_paths is 6 times that of gt_paths
    # assert len(input_paths) == 6 * len(gt_paths), (
    #     f'{input_key} dataset should have 6 times the number of images as the {gt_key} dataset: '
    #     f'{len(input_paths)} vs {len(gt_paths)}.')

    # # Create a mapping of GT filenames for quick lookup
    # gt_map = {}
    # for gt_path in gt_paths:
    #     basename, ext = osp.splitext(osp.basename(gt_path))
    #     gt_map[basename] = osp.join(gt_folder, gt_path)

    # paths = []

    # for input_path in input_paths:
    #     # Extract the GT name from the input filename
    #     input_basename, ext = osp.splitext(osp.basename(input_path))
    #     split_name = input_basename.split('_')
    #     assert len(split_name) >= 2, (
    #         f'Input filename {input_basename} does not have enough segments to extract GT name.')

    #     gt_name = f'{split_name[0]}_{split_name[1]}'
    #     assert gt_name in gt_map, (
    #         f'GT image {gt_name} is not found in {gt_key}_paths.')

    #     gt_path = gt_map[gt_name]
    #     paths.append(
    #         dict([(f'{input_key}_path', osp.join(input_folder, input_path)),
    #               (f'{gt_key}_path', gt_path)]))

    # return paths
    assert len(folders) == 3, (
        'The len of folders should be 3 with [input_folder, gt_folder, mask_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [input_key, gt_key, mask_key]. '
        f'But got {len(keys)}')

    input_folder, gt_folder, mask_folder = folders
    input_key, gt_key, mask_key = keys

    # Get the paths from folders
    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    mask_paths = list(scandir(mask_folder))

    # Check the relationship between files count
    assert len(input_paths) == 6 * len(gt_paths), (
        f'{input_key} dataset should have 6 times the number of images as the {gt_key} dataset: '
        f'{len(input_paths)} vs {len(gt_paths)}.')
    assert len(mask_paths) == len(gt_paths), (
        f'{mask_key} dataset should have the same number of images as {gt_key} dataset: '
        f'{len(mask_paths)} vs {len(gt_paths)}.')

    # Create a mapping for GT
    gt_map = {}
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        gt_map[basename] = osp.join(gt_folder, gt_path)

    # Create a mapping for mask
    mask_map = {}
    for mask_path in mask_paths:
        basename, ext = osp.splitext(osp.basename(mask_path))
        mask_map[basename] = osp.join(mask_folder, mask_path)

    paths = []
    for input_path in input_paths:
        input_basename, ext = osp.splitext(osp.basename(input_path))
        split_name = input_basename.split('_')
        assert len(split_name) >= 2, (
            f'Input filename {input_basename} does not have enough segments to extract GT and mask name.')

        # Construct gt_name and mask_name from the first two segments
        gt_name = f'{split_name[0]}_{split_name[1]}'
        mask_name = gt_name  # 同gt_name对应
        
        assert gt_name in gt_map, (
            f'GT image {gt_name} is not found in {gt_key}_paths.')
        assert mask_name in mask_map, (
            f'Mask image {mask_name} is not found in {mask_key}_paths.')

        gt_path = gt_map[gt_name]
        mask_path = mask_map[mask_name]

        paths.append(
            dict([(f'{input_key}_path', osp.join(input_folder, input_path)),
                  (f'{gt_key}_path', gt_path),
                  (f'{mask_key}_path', mask_path)]))

    return paths

def paired_paths_from_folder_ppr10k_val(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[dict]: Returned path list.
    """
    
    assert len(folders) == 3, (
        'The len of folders should be 3 with [input_folder, gt_folder, mask_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [input_key, gt_key, mask_key]. '
        f'But got {len(keys)}')

    input_folder, gt_folder, mask_folder = folders
    input_key, gt_key, mask_key = keys

    # Get the paths from folders
    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    mask_paths = list(scandir(mask_folder))
    # Check the relationship between files count
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} dataset should have the same number of images as the {gt_key} dataset: '
        f'{len(input_paths)} vs {len(gt_paths)}.')
    assert len(mask_paths) == len(gt_paths), (
        f'{mask_key} dataset should have the same number of images as {gt_key} dataset: '
        f'{len(mask_paths)} vs {len(gt_paths)}.')

    # Create a mapping for GT
    gt_map = {}
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        gt_map[basename] = osp.join(gt_folder, gt_path)

    # Create a mapping for mask
    mask_map = {}
    for mask_path in mask_paths:
        basename, ext = osp.splitext(osp.basename(mask_path))
        mask_map[basename] = osp.join(mask_folder, mask_path)

    paths = []
    for input_path in input_paths:
        input_basename, ext = osp.splitext(osp.basename(input_path))
        
        gt_name = input_basename
        mask_name = gt_name
        
        assert gt_name in gt_map, (
            f'GT image {gt_name} is not found in {gt_key}_paths.')
        assert mask_name in mask_map, (
            f'Mask image {mask_name} is not found in {mask_key}_paths.')

        gt_path = gt_map[gt_name]
        mask_path = mask_map[mask_name]

        paths.append(
            dict([(f'{input_key}_path', osp.join(input_folder, input_path)),
                  (f'{gt_key}_path', gt_path),
                  (f'{mask_key}_path', mask_path)]))

    return paths

def paired_paths_from_txt(txts, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(txts) == len(keys), (
        f'The len of txts should be the same between {txts} and {keys}')

    all_paths = []
    for txt in txts:
        all_paths.append([p.strip('\n') for p in open(txt, 'r').readlines()])
    for i, all_path in enumerate(all_paths[1:]):
        assert len(all_path) == len(all_paths[0]), (
            f'{keys[i+1]} and {keys[0]} datasets have different number of images: '
            f'{len(all_path)}, {len(all_path[0])}.')

    paths = []
    for single_paths in zip(*all_paths):
        for path in single_paths[1:]:
            assert osp.splitext(osp.basename(path))[0] == \
            osp.splitext(osp.basename(single_paths[0]))[0],\
            (f'{path} is not the same as ' f'{single_paths[0]}.')
        paths.append(
            dict([(f'{k}_path', p) for k, p in zip(keys, single_paths)]))

    # assert len(txts) == 2, (
    #     'The len of folders should be 2 with [input_folder, gt_folder]. '
    #     f'But got {len(txts)}')
    # assert len(keys) == 2, (
    #     'The len of keys should be 2 with [input_key, gt_key]. '
    #     f'But got {len(keys)}')
    # input_txt, gt_txt = txts
    # input_key, gt_key = keys

    # input_paths = [p.strip('\n') for p in open(input_txt, 'r').readlines()]
    # gt_paths = [p.strip('\n') for p in open(gt_txt, 'r').readlines()]

    # assert len(input_paths) == len(gt_paths), (
    #     f'{input_key} and {gt_key} datasets have different number of images: '
    #     f'{len(input_paths)}, {len(gt_paths)}.')
    # paths = []
    # for input_path, gt_path in zip(input_paths, gt_paths):
    #     assert osp.splitext(osp.basename(gt_path))[0] == \
    #         osp.splitext(osp.basename(input_path))[0],\
    #         (f'{input_path} is not the same as ' f'{gt_path}.')
    #     paths.append(
    #         dict([(f'{input_key}_path', input_path),
    #               (f'{gt_key}_path', gt_path)]))
    return paths

def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths
