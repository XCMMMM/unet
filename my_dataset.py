import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    # root--指向DRIVE数据集所在的根目录
    # train--为True载入training目录下的数据，为False载入test目录下的数据
    # transforms--针对数据的预处理方式
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        # 目录拼接，指向training目录或test目录
        data_root = os.path.join(root, "DRIVE", self.flag)
        # 断言判断路径是否存在
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # img_names--images文件夹下的每一张图片的名称（图片后缀为.tif）
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        # self.img_list--将图片名称和目录进行拼接，得到每一张图片的路径
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # self.manual--得到每一张标签文件的路径
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        # 判断每一个manual文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        # self.roi_mask--得到每一张mask文件的路径，mask文件黑白两色，白色部分是要进行分割的部分
        # self.flag--"training"/"test"
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        # 判断每一个mask文件是否存在
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
    
    # idx--传入索引
    # 返回对应索引的图片img和mask，这里的mask是ground truth。（注意这里的mask和原始数据集中的mask不一样）
    def __getitem__(self, idx):
        # 打开对应索引的图片，将图片转化为RGB图片
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 打开对应索引的标签文件，转化成灰度图
        manual = Image.open(self.manual[idx]).convert('L')
        # 由于之前前景像素值为255，背景为0，除以255后，前景为1，背景为0
        manual = np.array(manual) / 255
        # 打开对应索引的mask文件，转化成灰度图
        # 开始时感兴趣区域像素值为255，不感兴趣区域像素值为0
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 此时感兴趣区域像素值为0，不感兴趣区域像素值为255
        # 将不感兴趣区域的像素值设置为255，这样计算损失时可以将像素值为255区域的损失给忽略掉
        roi_mask = 255 - np.array(roi_mask)
        # 将manual和roi_mask进行相加，在利用np.clip方法为它设置一个上下限，上限255，下限0
        # 此时得到的mask，对于前景区域像素值为1，对于背景区域像素值为0，对于不感兴趣的区域像素值为255
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    # 将图片images和targets打包成一个batch
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

