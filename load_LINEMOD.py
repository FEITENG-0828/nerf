import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


# 定义一个函数，用于创建沿z轴平移的变换矩阵，t为平移的距离
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# 定义一个函数，用于创建绕x轴旋转的变换矩阵，phi是以度为单位的旋转角度，转换为弧度后用于三角函数计算
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi / 180. * np.pi), -np.sin(phi / 180. * np.pi), 0],
    [0, np.sin(phi / 180. * np.pi), np.cos(phi / 180. * np.pi), 0],
    [0, 0, 0, 1]]).float()

# 定义一个函数，用于创建绕y轴旋转的变换矩阵，th是以度为单位的旋转角度，转换为弧度后用于三角函数计算
rot_theta = lambda th: torch.Tensor([
    [np.cos(th / 180. * np.pi), 0, -np.sin(th / 180. * np.pi), 0],
    [0, 1, 0, 0],
    [np.sin(th / 180. * np.pi), 0, np.cos(th / 180. * np.pi), 0],
    [0, 0, 0, 1]]).float()


# 该函数用于根据给定的球坐标参数（方位角theta、仰角phi和半径radius）生成相机到世界坐标系的变换矩阵（c2w）
def pose_spherical(theta, phi, radius):
    # 先创建沿z轴平移radius距离的变换矩阵
    c2w = trans_t(radius)
    # 绕x轴旋转phi角度，将旋转矩阵与之前的变换矩阵相乘，实现复合变换
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    # 绕y轴旋转theta角度，同样与前面的结果相乘
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    # 再乘以一个固定的变换矩阵，可能用于调整坐标系的方向等特定需求
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


# 该函数用于加载LINEMOD数据集的数据，包括图像、姿态等信息
def load_LINEMOD_data(basedir, half_res=False, testskip=1):
    # 定义数据集的分割，通常有训练集、验证集和测试集
    splits = ['train', 'val', 'test']
    metas = {}
    # 遍历每个分割，读取对应的json文件，其中包含了数据集相关的元信息（如图像路径、姿态等），并存储在metas字典中
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    # 再次遍历每个分割
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # 根据是否是训练集或者testskip的值来决定读取数据时的间隔（skip值），如果是训练集或者testskip为0，则间隔为1，即读取所有数据，否则按testskip的值间隔读取
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # 遍历当前分割中的每一帧数据（按设定的间隔）
        for idx_test, frame in enumerate(meta['frames'][::skip]):
            fname = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            # 读取图像数据
            imgs.append(imageio.imread(fname))
            # 获取当前帧对应的姿态变换矩阵
            poses.append(np.array(frame['transform_matrix']))

        # 将图像数据转换为浮点数类型，并归一化到 [0, 1] 区间，同时保留RGBA四个通道（如果有）
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        # 将姿态变换矩阵转换为浮点数类型的numpy数组
        poses = np.array(poses).astype(np.float32)
        # 记录当前分割数据的起始索引（用于后续划分数据集索引）
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 根据counts列表生成每个分割对应的索引范围，用于划分训练、验证、测试集的索引
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # 将所有分割的图像数据沿第一个维度（样本维度）进行拼接
    imgs = np.concatenate(all_imgs, 0)
    # 将所有分割的姿态数据沿第一个维度进行拼接
    poses = np.concatenate(all_poses, 0)

    # 获取第一张图像的高度和宽度
    H, W = imgs[0].shape[:2]
    # 获取相机内参矩阵中的焦距值（这里取第一个元素，假设内参矩阵是对称的情况）
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    # 获取相机内参矩阵（完整的）
    K = meta['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")

    # 生成一系列用于渲染的相机姿态，通过在一定角度范围内调用pose_spherical函数生成不同角度的相机到世界坐标系的变换矩阵，并堆叠成一个张量
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # 如果half_res为True，表示要将图像分辨率减半
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        # 遍历每一张图像，使用cv2.resize进行下采样，将图像尺寸减半，并存入新的数组中
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # 获取训练集和测试集中的近裁剪面距离的最小值，并向下取整
    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    # 获取训练集和测试集中的远裁剪面距离的最大值，并向上取整
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far