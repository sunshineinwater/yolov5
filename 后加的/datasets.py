import numpy as np
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective


class LoadImageFromArray:
    mode = "image"

    def __init__(self, images: np.ndarray, img_size=640, stride=32, auto=True):
        """
        @images: cv2BGR格式数据, shape = [图片数量,宽度,高度,深度] or [宽度,高度,深度]
                 该类有自动转为torch图片格式的代码
        """
        self.images = images
        self.img_size = img_size
        self.stride = stride
        self.auto = auto

        #
        shape = images.shape
        # 图片总数 nf
        if len(shape) == 4:
            self.nf = shape[0]
        elif len(shape) == 3:
            # 增加维度
            self.images = images[np.newaxis, :, :, :]
            self.nf = 1
        else:
            raise IOError("image维度错误")
        #

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # 达到上限,停止迭代
        if self.count == self.nf:
            raise StopIteration
        # 提取对应image
        img0 = self.images[self.count]
        self.count += 1

        # 图片处理
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

        # 组合参数
        s = f'image {self.count}/{self.nf} : '  # s: 当前图片索引提示 - 字符串

        # path,格式化后的图片,原始图片,摄像头,提示字符串
        return f"image{self.count - 1}.jpg", img, img0, None, s

    def __len__(self):
        return self.nf
