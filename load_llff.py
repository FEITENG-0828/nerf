import numpy as np
import os
import imageio

# 用于缩小图像尺寸的函数，支持按比例因子（factors）缩小或者指定分辨率（resolutions）缩小
def _minify(basedir, factors=[], resolutions=[]):
    # 标记是否需要加载（即是否需要进行图像缩小操作），初始化为False
    needtoload = False
    # 遍历比例因子列表，如果对应比例因子的图像目录不存在，则需要进行加载操作
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    # 遍历分辨率列表，如果对应分辨率的图像目录不存在，则需要进行加载操作
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    # 如果不需要加载（即所有对应目录都已存在），直接返回
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    # 获取原始图像所在目录
    imgdir = os.path.join(basedir, 'images')
    # 获取目录下所有图像文件的路径，筛选出常见的图像格式文件（JPG、jpg、png、jpeg、PNG）
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    # 获取当前工作目录，用于后续切换目录后能返回原目录
    wd = os.getcwd()

    # 遍历比例因子和分辨率列表
    for r in factors + resolutions:
        # 如果r是整数，说明是按比例因子缩小，设置相应的目录名和调整参数格式
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        # 如果r是列表（表示分辨率），则设置相应的目录名和调整参数格式（宽x高）
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        # 构建目标图像目录路径
        imgdir = os.path.join(basedir, name)
        # 如果目标目录已存在，跳过本次循环（无需重复创建和处理）
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        # 创建目标图像目录
        os.makedirs(imgdir)
        # 复制原始图像目录下的所有文件到目标目录
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        # 获取图像文件的扩展名
        ext = imgs[0].split('.')[-1]
        # 构建用于调整图像尺寸和格式的命令参数
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        # 切换到目标图像目录
        os.chdir(imgdir)
        # 执行图像尺寸调整和格式转换命令
        check_output(args, shell=True)
        # 切换回原始工作目录
        os.chdir(wd)

        # 如果原始图像扩展名不是png，删除目标目录下的原格式图像文件（因为已转换为png格式了）
        if ext!= 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


# 用于加载数据的函数，可根据指定条件（如比例因子、宽度、高度等）加载图像、姿态和边界信息
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    # 加载包含姿态和边界信息的numpy数组文件
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # 提取姿态信息并调整维度顺序，使其符合特定的格式要求（这里可能与后续的计算和处理相关）
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    # 提取边界信息并调整维度顺序
    bds = poses_arr[:, -2:].transpose([1, 0])

    # 获取第一张图像的路径（从基于目录的图像目录中筛选出合适的图像文件）
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # 获取该图像的形状（高度、宽度、通道数等信息）
    sh = imageio.imread(img0).shape

    # 用于拼接在图像目录名后的后缀字符串，初始为空
    sfx = ''

    # 如果指定了比例因子
    if factor is not None:
        sfx = '_{}'.format(factor)
        # 调用_minify函数按比例因子缩小图像
        _minify(basedir, factors=[factor])
        factor = factor
    # 如果指定了高度
    elif height is not None:
        # 计算比例因子（根据原始高度和指定高度的比例）
        factor = sh[0] / float(height)
        # 计算调整后的宽度
        width = int(sh[1] / factor)
        # 调用_minify函数按指定分辨率缩小图像
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    # 如果指定了宽度
    elif width is not None:
        # 计算比例因子（根据原始宽度和指定宽度的比例）
        factor = sh[1] / float(width)
        # 计算调整后的高度
        height = int(sh[0] / factor)
        # 调用_minify函数按指定分辨率缩小图像
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    # 如果都没指定，比例因子设为1（即不进行缩放）
    else:
        factor = 1

    # 构建最终的图像目录路径（根据后缀情况）
    imgdir = os.path.join(basedir, 'images' + sfx)
    # 如果图像目录不存在，打印提示信息并返回（无法加载数据）
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    # 获取图像目录下所有符合格式要求的图像文件路径
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg')
                or f.endswith('png')]
    # 检查图像数量和姿态信息的维度是否匹配，如果不匹配则打印错误信息并返回
    if poses.shape[-1]!= len(imgfiles):
        print('Mismatch between imgs {} and poses {}!!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    # 将图像的高度和宽度信息更新到姿态信息中（前两维对应图像尺寸相关，这里可能与相机内参等相关）
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # 根据比例因子调整姿态信息中与深度相关的部分（可能涉及相机坐标变换等）
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # 如果不需要加载图像数据，只返回姿态和边界信息
    if not load_imgs:
        return poses, bds

    # 定义一个读取图像的函数，针对png图像设置忽略gamma校正，其他图像正常读取
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    # 读取所有图像文件，转换为浮点数类型，并归一化到 [0, 1] 区间，同时只保留前三个通道（去除可能的alpha通道等），并堆叠成一个多维数组
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


# 用于将向量归一化（使其长度为1）的函数
def normalize(x):
    return x / np.linalg.norm(x)


# 根据给定的视线方向（z）、向上方向（up）和相机位置（pos）构建视图矩阵（相机到世界坐标系的变换矩阵）
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


# 将世界坐标系中的点转换到相机坐标系下的函数
def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


# 计算一组姿态的平均姿态（通过平均位置、平均方向等方式来估计一个大致的平均相机到世界坐标系的变换矩阵）
def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


# 生成螺旋形渲染路径上的相机姿态的函数，用于渲染多视图场景
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    # 在给定的角度范围内均匀采样，生成螺旋路径上的不同相机姿态
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


# 重新调整一组姿态的中心位置，使其以平均姿态为中心（可能用于数据预处理，统一坐标系统等）
def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


# 将一组姿态进行球面化处理的函数，改变姿态的分布使其更符合球面分布（常用于特定的场景表示和渲染需求）
def spherify_poses(poses, bds):
    # 一个将3x4的姿态矩阵转换为4x4齐次坐标矩阵的lambda函数（添加最后一行的齐次坐标表示）
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    # 计算一组光线（由起点和方向表示）到某一点的最小距离的函数（通过一些线性代数运算来求解）
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    # 计算光线到某一点的最小距离
    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    # 对姿态进行变换，使其以计算出的中心和方向为参考进行重置
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    # 生成一组在球面上均匀分布的新姿态（通过在圆周上均匀采样位置等方式构建相机姿态）
    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    # 将新生成的姿态与原始姿态的一些相关信息（可能是图像尺寸等）进行合并，保持维度等一致性
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)],
                                 -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    """
    加载LLFF数据集的函数，可根据不同参数进行数据预处理、姿态调整以及生成用于渲染的相机姿态等操作，并返回相关数据。

    参数:
    - basedir: 数据集所在的基础目录路径。
    - factor: 图像下采样的因子，默认为8，即按8倍缩小原始图像。
    - recenter: 是否重新调整姿态的中心位置，默认为True。
    - bd_factor: 用于边界缩放的因子，若为None则不进行边界缩放。
    - spherify: 是否对姿态进行球面化处理，默认为False。
    - path_zflat: 是否生成平面化的渲染路径（在z轴方向上进行特定处理），默认为False。

    返回值:
    - images: 处理后的图像数据，维度为 (num_images, height, width, channels)，数据类型为float32。
    - poses: 相机姿态数据，维度为 (num_images, 3, 4)，数据类型为float32。
    - bds: 深度边界数据，维度和具体含义根据内部处理而定，数据类型为float32。
    - render_poses: 用于渲染的相机姿态数据，维度和格式与poses类似，数据类型为float32。
    - i_test: 用于测试的视图索引（通过某种距离计算得出的最小距离对应的索引）。
    """

    # 调用_load_data函数加载数据，factor=8表示默认按8倍缩小原始图像（可根据参数调整）
    poses, bds, imgs = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    # 对姿态矩阵进行维度调整和旋转矩阵顺序的修正，将原本姿态矩阵中不符合后续处理要求的维度顺序进行调整
    # 这里将姿态矩阵的维度顺序调整为符合特定的坐标系统和矩阵运算要求的格式，先交换一些行列，再将最后一维（样本维度）移到最前面
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    # 根据bd_factor的值决定是否对数据进行缩放，如果bd_factor为None，则缩放因子sc设为1，即不缩放；
    # 否则，计算缩放因子sc，使得深度边界数据（bds）根据最小边界值和bd_factor进行缩放，同时相机姿态中的位置信息也相应缩放
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    # 如果recenter参数为True，调用recenter_poses函数重新调整姿态的中心位置，使其以平均姿态为中心，常用于数据预处理，统一坐标系统等
    if recenter:
        poses = recenter_poses(poses)

    # 如果spherify参数为True，调用spherify_poses函数对姿态进行球面化处理，改变姿态的分布使其更符合球面分布，常用于特定的场景表示和渲染需求
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 计算所有姿态的平均姿态，得到一个大致的相机到世界坐标系的变换矩阵，后续用于生成渲染路径等操作的参考
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        # 计算所有姿态的平均向上方向向量（这里通过对姿态矩阵中对应向上方向的向量进行求和后归一化得到），
        # 该向上方向向量用于后续构建相机姿态时确定相机的“上”方向
        up = normalize(poses[:, :3, 1].sum(0))

        # 确定一个合理的“聚焦深度”（焦点深度），通过结合深度边界的最小值和最大值，按照一定权重计算得出，
        # 这个深度值在生成螺旋渲染路径等操作中可能与相机焦距等相关参数配合使用
        close_depth, inf_depth = bds.min() *.9, bds.max() * 5.
        dt =.75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        # 设置螺旋路径的一些参数，包括收缩因子（用于调整螺旋半径等）、z方向的偏移量（控制螺旋在z轴方向的变化情况）
        # 计算螺旋路径的半径（基于姿态中相机位置的绝对值的百分位数来确定，这里取90%分位数，可能是为了排除一些异常值影响）
        shrink_factor =.8
        zdelta = close_depth *.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T，获取相机在世界坐标系中的位置信息
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2

        # 如果path_zflat参数为True，进行特定的平面化处理，例如调整螺旋路径的中心位置在z轴方向的偏移，
        # 并将螺旋路径在z轴方向的半径设为0（使其在z轴方向相对“扁平”），同时减少视图数量和旋转次数
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth *.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        # 调用render_path_spiral函数，根据前面计算得到的参数（平均姿态、向上方向、螺旋半径、焦距等）
        # 生成螺旋形渲染路径上的相机姿态，这些姿态用于后续的场景渲染等操作
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    # 将生成的渲染姿态数据转换为指定的float32数据类型
    render_poses = np.array(render_poses).astype(np.float32)

    # 再次计算所有姿态的平均姿态（这里可能是为了后续其他用途或者再次确认平均姿态情况）
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    # 计算每个姿态与平均姿态在位置上的距离平方和，通过找到最小距离对应的索引，确定用于测试的视图索引（可能用于划分训练集、测试集等操作）
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    # 将图像数据和姿态数据都转换为float32数据类型（确保数据类型的一致性，方便后续处理）
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test