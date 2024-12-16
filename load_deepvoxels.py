import os
import numpy as np
import imageio 


# 函数load_dv_data用于加载DeepVoxels数据集相关的数据
# scene：       表示要加载的具体场景名称，默认值为'cube'，用于确定在数据集中具体加载哪个场景的数据
# basedir：     数据集所在的基础目录路径，函数会在该目录下查找对应场景不同数据集分割（训练、验证、测试）的相关文件
# testskip：    整数类型参数，用于控制在测试集数据读取时的采样间隔，默认值为8，表示每隔8帧读取一帧数据
def load_dv_data(scene='cube', basedir='/data/deepvoxels', testskip=8):
    
    # 解析相机内参文件，获取相机内参、网格重心、缩放比例、近平面距离以及世界到相机姿态是否存在等信息，并根据目标边长调整内参
    # 参数filepath：        相机内参文件的完整路径，从中读取相机内参相关的数据
    # 参数trgt_sidelength： 目标边长，用于根据当前图像尺寸与目标尺寸的比例来调整相机内参中的一些参数，比如焦距、光心坐标等
    # 参数invert_y：        布尔类型参数，用于决定是否对y轴方向的焦距进行反转，默认值为False，通常在不同坐标系或图像方向需求下可能会设置为True
    def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
        # 获取相机内参
        with open(filepath, 'r') as file:
            # 从文件第一行读取焦距f、光心x坐标cx、光心y坐标cy，只取前三个值（这里假设文件每行数据格式固定且可能有多余数据），并将它们转换为浮点数类型
            f, cx, cy = list(map(float, file.readline().split()))[:3]
            # 从文件第二行读取网格重心坐标（以空格分隔的浮点数形式存储在文件中），并转换为NumPy数组形式
            grid_barycenter = np.array(list(map(float, file.readline().split())))
            # 从文件第三行读取近平面距离，转换为浮点数类型
            near_plane = float(file.readline())
            # 从文件第四行读取缩放比例，转换为浮点数类型
            scale = float(file.readline())
            # 从文件第五行读取图像的高度和宽度，转换为浮点数类型
            height, width = map(float, file.readline().split())

            try:
                # 尝试从文件第六行读取世界到相机姿态相关的标识（可能是整数形式表示是否存在等情况），如果读取成功则赋值给world2cam_poses变量
                world2cam_poses = int(file.readline())
            except ValueError:
                # 如果读取失败（比如该行数据格式不符合预期等情况），则将world2cam_poses设为None，表示未获取到有效标识
                world2cam_poses = None

        # 如果world2cam_poses为None，将其转换为布尔类型的False，表示不存在世界到相机的姿态信息（根据后续逻辑的约定）
        if world2cam_poses is None:
            world2cam_poses = False

        # 确保world2cam_poses为布尔类型，将其转换为明确的布尔值（如果之前不是布尔类型的话）
        world2cam_poses = bool(world2cam_poses)

        print(cx, cy, f, height, width)

        # 根据目标边长与当前图像宽度的比例，调整光心x坐标cx，使其对应到目标边长下的坐标值
        cx = cx / width * trgt_sidelength
        # 同样根据目标边长与当前图像高度的比例，调整光心y坐标cy，使其对应到目标边长下的坐标值
        cy = cy / height * trgt_sidelength
        # 根据目标边长与当前图像高度的比例，调整焦距f，使其对应到目标边长下的焦距值
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            # 如果invert_y为True，对y轴方向的焦距fy进行反转（变为负的焦距值），通常可能用于符合特定的坐标系要求或图像坐标方向的设定
            fy = -f
        else:
            fy = f

        # 构建完整的相机内参矩阵，采用齐次坐标形式，是一个4x4的NumPy数组，包含了调整后的焦距、光心坐标等信息，用于后续的坐标变换、投影等操作与相机成像相关的计算
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])

        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses

    # 函数load_pose用于加载相机姿态数据文件，将文件中的数据解析为4x4的NumPy数组形式的相机姿态矩阵，并转换为float32类型
    # 参数filename：相机姿态数据文件的完整路径，文件中存储了相机姿态相关的数值信息（格式需符合函数内的解析逻辑）
    def load_pose(filename):
        assert os.path.isfile(filename)
        # 读取文件内容，按空格分割成字符串列表，每个字符串表示一个数值
        nums = open(filename).read().split()
        # 将字符串列表中的每个元素转换为浮点数，并重新调整形状为4x4的NumPy数组，最后转换为float32类型，得到相机姿态矩阵
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    # 设置图像的高度和宽度，这里假设所有图像在该数据集中尺寸固定为512x512像素
    H = 512
    W = 512
    # 构建DeepVoxels数据集对应场景的基础路径，用于后续查找该场景下训练集相关的文件（如内参文件、姿态文件、图像文件等）
    deepvoxels_base = '{}/train/{}/'.format(basedir, scene)

    # 调用parse_intrinsics函数解析训练集的相机内参文件，获取相机内参、网格重心、缩放比例、近平面距离以及世界到相机姿态相关信息，传入目标边长H（即图像高度，这里假设目标边长与图像高度一致用于调整内参）
    full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses = parse_intrinsics(os.path.join(deepvoxels_base, 'intrinsics.txt'), H)
    print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
    # 获取相机内参矩阵中的焦距值（这里取内参矩阵左上角元素，通常假设fx和fy相等情况下），用于后续可能的计算或操作
    focal = full_intrinsic[0, 0]
    print(H, W, focal)

    # 函数dir2poses用于从给定目录下加载所有相机姿态文件，并进行坐标变换等处理，最终返回处理后的相机姿态矩阵数组
    def dir2poses(posedir):
        # 通过列表推导式加载目录下所有以'txt'结尾的相机姿态文件，调用load_pose函数解析每个文件内容为相机姿态矩阵，然后将这些矩阵沿第0维堆叠起来，形成一个三维的NumPy数组（维度为(num_poses, 4, 4)，num_poses表示姿态文件数量）
        poses = np.stack([load_pose(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0)
        # 定义一个固定的坐标变换矩阵transf，用于对相机姿态进行变换，这里可能是根据数据集的坐标系特点或者后续处理需求进行的一种坐标调整，比如对某些坐标轴进行反转等操作
        transf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.],
        ])
        # 将每个相机姿态矩阵与坐标变换矩阵transf进行矩阵乘法运算，实现对相机姿态的坐标变换操作，以符合特定的坐标系或处理要求
        poses = poses @ transf
        # 截取姿态矩阵的前3行和前4列，将其转换为只包含旋转和平移信息的相机姿态矩阵（去掉了齐次坐标表示中的最后一行），并转换为float32类型，使其维度变为(num_poses, 3, 4)，方便后续与其他计算对接
        poses = poses[:, :3, :4].astype(np.float32)
        return poses

    # 构建训练集相机姿态文件所在的目录路径，用于后续调用dir2poses函数加载训练集的相机姿态数据
    posedir = os.path.join(deepvoxels_base, 'pose')
    # 调用dir2poses函数加载训练集的相机姿态数据
    poses = dir2poses(posedir)
    # 构建测试集相机姿态文件所在的目录路径，调用dir2poses函数加载测试集的相机姿态数据，并按照testskip指定的采样间隔进行采样读取
    testposes = dir2poses('{}/test/{}/pose'.format(basedir, scene))
    testposes = testposes[::testskip]
    # 构建验证集相机姿态文件所在的目录路径，调用dir2poses函数加载验证集的相机姿态数据，并按照testskip指定的采样间隔进行采样读取
    valposes = dir2poses('{}/validation/{}/pose'.format(basedir, scene))
    valposes = valposes[::testskip]

    # 获取训练集图像文件列表，筛选出以'png'结尾的文件，即获取所有训练集的图像文件名，这里假设图像文件都存储在'rgb'目录下且文件格式为'png'
    imgfiles = [f for f in sorted(os.listdir(os.path.join(deepvoxels_base, 'rgb'))) if f.endswith('png')]
    # 通过列表推导式读取训练集的所有图像文件，使用imageio库的imread函数读取图像数据，将图像像素值归一化（除以255.），然后将所有图像数据沿第0维堆叠起来，形成一个三维的NumPy数组（维度为(num_imgs, H, W, channels)，num_imgs表示图像数量，channels表示图像通道数），最后转换为float32类型，得到训练集的图像数据数组
    imgs = np.stack([imageio.imread(os.path.join(deepvoxels_base, 'rgb', f)) / 255. for f in imgfiles], 0).astype(np.float32)

    # 构建测试集图像文件所在的目录路径，获取测试集图像文件列表，筛选出以'png'结尾的文件，即获取所有测试集的图像文件名
    testimgd = '{}/test/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith('png')]
    # 通过列表推导式读取测试集的图像文件（按照testskip指定的采样间隔进行读取），进行图像数据读取、归一化以及堆叠操作，得到测试集的图像数据数组，与训练集图像数据处理方式类似
    testimgs = np.stack([imageio.imread(os.path.join(testimgd, f)) / 255. for f in imgfiles[::testskip]], 0).astype(np.float32)

    # 构建验证集图像文件所在的目录路径，获取验证集图像文件列表，筛选出以'png'结尾的文件，即获取所有验证集的图像文件名
    valimgd = '{}/validation/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(valimgd)) if f.endswith('png')]
    # 通过列表推导式读取验证集的图像文件（按照testskip指定的采样间隔进行读取），进行图像数据读取、归一化以及堆叠操作，得到验证集的图像数据数组，同样与训练集图像数据处理方式类似
    valimgs = np.stack([imageio.imread(os.path.join(valimgd, f)) / 255. for f in imgfiles[::testskip]], 0).astype(np.float32)

    # 将训练集、验证集、测试集的图像数据列表合并为一个列表，方便后续进行合并等操作
    all_imgs = [imgs, valimgs, testimgs]
    # 创建一个计数列表counts，初始值为[0]，用于记录每个数据集分割的图像数量的累积情况，方便后续进行索引操作等，首先添加0表示起始计数
    counts = [0] + [x.shape[0] for x in all_imgs]
    # 对计数列表counts进行累积求和操作，得到每个数据集分割对应的图像数据索引范围的累积计数，例如counts[1]表示训练集图像数量，counts[2]表示训练集和验证集图像数量总和等
    counts = np.cumsum(counts)
    # 根据累积计数列表counts生成索引列表i_split，用于划分不同数据集分割对应的图像数据索引范围，例如i_split[0]对应训练集图像数据的索引范围，i_split[1]对应验证集的索引范围，i_split[2]对应测试集的索引范围
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    # 将所有分割的图像数据沿第0维进行合并，形成一个总的图像数据数组，包含了训练集、验证集和测试集的所有图像数据
    imgs = np.concatenate(all_imgs, 0)
    # 将所有分割的相机姿态数据沿第0维进行合并，得到总的相机姿态数据数组，包含了训练集、验证集和测试集的所有相机姿态信息
    poses = np.concatenate([poses, valposes, testposes], 0)

    # 将测试集的相机姿态数据赋值给render_poses，通常可能用于后续基于测试集姿态进行场景渲染等相关操作（具体取决于整个项目的功能需求）
    render_poses = testposes

    print(poses.shape, imgs.shape)

    return imgs, poses, render_poses, [H, W, focal], i_split