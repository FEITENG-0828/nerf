import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

# 沿z轴平移的齐次坐标变换矩阵
trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

# 创建绕x轴旋转的齐次坐标变换矩阵
rot_phi = lambda phi : torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# 创建绕y轴旋转的齐次坐标变换矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

# 根据给定的球坐标参数（水平角度theta、垂直角度phi和半径radius）生成相机到世界坐标系的变换矩阵（c2w，camera-to-world）
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

# 函数load_blender_data用于加载数据集相关的数据
# basedir：数据集所在的基础目录路径，函数会在该目录下查找不同数据集分割（训练、验证、测试）对应的JSON配置文件等数据
# half_res：布尔类型参数，用于决定是否将图像分辨率减半，默认值为False，即不进行减半操作
# testskip：整数类型参数，用于控制在测试集数据读取时的采样间隔，默认值为1，表示逐帧读取；若大于1，则会每隔testskip帧读取一帧数据
def load_blender_data(basedir, half_res=False, testskip=1):
    # 定义数据集分割的名称列表，包含'train'（训练集）、'val'（验证集）、'test'（测试集）
    splits = ['train', 'val', 'test']
    # 创建一个空字典metas，用于存储各个数据集分割对应的元数据（从JSON配置文件中读取的数据）
    metas = {}
    # 循环读取每个分割对应的JSON配置文件，并将解析后的JSON数据存储到metas字典中
    # 配置文件的文件名格式为'transforms_{}.json'，其中{}会被s的具体值（如'train'、'val'、'test'）替换
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []           # 存储所有的图像数据（后续会将不同分割的图像数据合并到这里）
    all_poses = []          # 存储所有的相机姿态数据（同样会合并不同分割的相机姿态信息）
    counts = [0]            # 创建一个计数列表counts，初始值为[0]，用于记录每个数据集分割的图像数量的累积情况，方便后续进行索引操作等

    # 循环遍历每个数据集分割（train、val、test）
    for s in splits:
        meta = metas[s]     # 获取当前分割元数据
        imgs = []           # 存储当前分割下的图像数据
        poses = []          # 存储当前分割下的相机姿态数据

        # 根据数据集分割情况决定数据读取的采样间隔
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # 循环读取当前分割下的每一帧数据（根据设置的采样间隔skip进行读取）
        for frame in meta['frames'][::skip]:
            # 构建当前帧对应的图像文件的完整路径，文件名格式为frame['file_path'] + '.png'
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # 使用imageio库的imread函数读取图像文件，并将读取到的图像数据添加到imgs列表中
            imgs.append(imageio.imread(fname))
            # 将当前帧对应的相机姿态数据（以NumPy数组形式存储在frame['transform_matrix']中）添加到poses列表中
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)       # 将当前分割下读取到的所有图像数据转换为float32类型，并进行归一化处理（除以255.），这里保留了图像的所有4个通道（RGBA）
        poses = np.array(poses).astype(np.float32)              # 将当前分割下读取到的所有相机姿态数据转换为float32类型的NumPy数组
        counts.append(counts[-1] + imgs.shape[0])               # 更新计数列表counts，记录当前分割的图像数量累积情况，即在上一次累积数量的基础上加上当前分割的图像数量（imgs.shape[0]表示当前分割图像数据的数量）
        all_imgs.append(imgs)                                   # 将当前分割下的图像数据列表imgs添加到总的图像数据列表all_imgs中
        all_poses.append(poses)                                 # 将当前分割下的相机姿态数据列表poses添加到总的相机姿态数据列表all_poses中

    # 根据计数列表counts生成索引列表i_split，用于划分不同数据集分割对应的图像数据索引范围
    # 例如，i_split[0]对应训练集图像数据的索引范围，i_split[1]对应验证集的索引范围，i_split[2]对应测试集的索引范围
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # 将所有分割的图像数据沿第0维进行合并，形成一个总的图像数据数组
    imgs = np.concatenate(all_imgs, 0)
    # 同样将所有分割的相机姿态数据沿第0维进行合并，得到总的相机姿态数据数组
    poses = np.concatenate(all_poses, 0)

    # 获取图像数据中第一张图像的高度和宽度（因为假设所有图像尺寸一致），用于后续计算相机参数等操作
    H, W = imgs[0].shape[:2]
    # 从元数据（以训练集的元数据为例，因为假设各分割的相机角度参数一致）中获取相机水平视角角度（单位为弧度）
    camera_angle_x = float(meta['camera_angle_x'])
    # 根据相机水平视角角度和图像宽度计算相机焦距，这里使用了简单的三角函数关系来推导焦距计算公式
    focal =.5 * W / np.tan(.5 * camera_angle_x)

    # 通过pose_spherical函数生成一系列渲染姿态（render_poses），用于后续渲染场景等操作
    # 这里生成了40个不同水平角度（从-180到180均匀分布）的相机到世界坐标系的变换矩阵，垂直角度固定为-30.0，半径固定为4.0
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # 如果half_res参数为True，表示要将图像分辨率减半
    if half_res:
        # 将图像高度和宽度都除以2，实现分辨率减半操作
        H = H // 2
        W = W // 2
        # 同时相机焦距也相应减半，以适应分辨率的变化
        focal = focal / 2.

        # 创建一个新的数组imgs_half_res，用于存储分辨率减半后的图像数据，初始化为全0数组，形状根据减半后的图像尺寸和通道数确定
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # 循环遍历所有图像数据，对每张图像使用cv2库的resize函数进行分辨率减半操作（采用区域插值法INTER_AREA），并将结果存储到imgs_half_res数组中
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # 将处理后的图像数据（分辨率减半后的）赋值给imgs变量，覆盖原来的图像数据
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()  （这行代码被注释掉了，可能原本有使用TensorFlow进行图像缩放的意图，但当前代码中未启用）

    # 最后返回加载和处理后的图像数据、相机姿态数据、渲染姿态数据、图像尺寸及焦距信息以及数据集分割索引范围信息
    return imgs, poses, render_poses, [H, W, focal], i_split