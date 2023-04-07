import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    # target--我们的标签文件，背景为0，前景为1，不感兴趣区域255
    dice_target = target.clone()
    # ignore_index默认为-100，由于我们将不感兴趣区域设置为255，所以ignore_index=255满足if语句
    if ignore_index >= 0:
        # 通过torch.eq方法寻找所有像素值等于255的位置
        ignore_mask = torch.eq(target, ignore_index)
        # 将像素值为255的位置全部替换成0
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        # one_hot编码转化后，多了一个channel维度，背景维0->10，前景维1->01，两个channel
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        # 再将之间像素值替换为0的位置全部替换回来
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
    # torch中把channel维度放在索引为1的位置
    return dice_target.permute(0, 3, 1, 2)

# x--针对某一个类别的预测概率矩阵
# target--针对某一个类别的ground truth
# ignore_index--哪些数值区域需要被忽略
def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        # 通过x_i就能取出当前batch中第i张图片对应某一类别的预测概率矩阵，并进行reshape
        x_i = x[i].reshape(-1)
        # 通过t_i就能取出当前batch中第i张图片的ground truth，并进行reshape
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index=255的区域
            roi_mask = torch.ne(t_i, ignore_index)
            # 将预测值中感兴趣的区域以及target中感兴趣的区域提取出来
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        # 将两个向量进行内积操作，即矩阵对应元素进行相乘再相加，对应于dice分子部分的计算
        inter = torch.dot(x_i, t_i)
        # 对应于分母部分的计算
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        # 如果sets_sum为0，则x_i和t_i的每一个元素都为0，此时预测都是对的
        if sets_sum == 0:
            sets_sum = 2 * inter
        # epsilon--防止分母为0的情况
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    # 将针对当前batch中的某个类别的所有dice_coeff的数值之和除以batch_size
    # 得到针对每张图片的对应某个类别的dice_coeff的均值
    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    # 遍历每一个channel，即遍历每一个类别的预测值和target，计算dice_coeff，将所有channel的dice_coeff相加
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)
    # 将dice除以类别数（channel），就能得到所有类别的dice的均值
    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    # 对预测值x在channel方向进行softmax处理，就能得到每个像素针对每个类别的概率
    x = nn.functional.softmax(x, dim=1)
    # multiclass--是否针对每一个类别进行dice loss计算，这里传入为True
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)
