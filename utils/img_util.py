import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
from PIL import Image, ImageFilter
import random

###manage groundtruth
def UnMaskFilterGaussian(img, Rad, Perc, Thr):
    single_c = False
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:,:,0]
        single_c = True
    pimg = Image.fromarray(img)
    dimg = pimg.filter(ImageFilter.UnsharpMask(radius=Rad, percent=Perc, threshold=Thr))
    rimg = np.array(dimg)
    if single_c:
        rimg = np.expand_dims(rimg, axis=2)
    return rimg

def UnMaskFilterBilateral(img, d, sigmacolor, sigmaspace, Perc, Ther):
    single_c = False
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:,:,0]
        single_c = True
    blurred = cv2.bilateralFilter(img, d, sigmacolor, sigmaspace)
    sharpened = img + (img - blurred) * Perc / 100.0
    # sharpened = blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if Ther > 0:
        low_contrast_mask = np.absolute(img - blurred) < Ther
        np.copyto(sharpened, img, where=low_contrast_mask)
    if single_c:
        sharpened = np.expand_dims(sharpened, axis=2)
    return sharpened
    #dimg = Image.fromarray(sharpened)
    #return dimg


## sharpen by xiaoming
def UnMaskFilterGD(img):
    if random.random() > 0.3: #
        Rad = random.randint(3,15)# 3~15
        Perc = random.randint(30,110) #110
        # Thr = random.randint(0,5) #0~3
        Thr = 0
        return UnMaskFilterGaussian(img, Rad, Perc, Thr)
    else:
        d = random.randint(3,9)
        sigmacolor = random.randint(150,300)
        sigmaspace = random.randint(150,300)
        Perc = random.randint(100,210) #110
        # Ther = random.randint(0,5)
        Ther = 0
        return UnMaskFilterBilateral(img, d, sigmacolor, sigmaspace, Perc, Ther)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def imfrombytes(content, flag='color', resize=False, float32=False, sharpen=False, img_size=(300,300)):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    
    if sharpen:
        img = UnMaskFilterGD(img)
    if resize:
        # img = cv2.resize(img, (256, 256))
        # print('before resize大小: ', img.shape)
        # print('before resize 位数: ', img.dtype)
        # height, width = img.shape[:2]
        img = cv2.resize(img, img_size)
    if float32:
        if flag == 'unchanged':
            img = img.astype(np.float32) / 65535.
        else:
            img = img.astype(np.float32) / 255.
    if flag == 'grayscale':
        img = img[:,:,np.newaxis]
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    # return cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    file_path = file_path.rsplit('.', 1)[0] + '.png'  # 替换后缀为 .png
    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # PNG 压缩级别
    

    # 保存文件
    return cv2.imwrite(file_path, img, params)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]
