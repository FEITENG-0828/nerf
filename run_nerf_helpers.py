import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# x 与 y 差值的平方的均值
img2mse = lambda x, y : torch.mean((x - y) ** 2)
# 计算峰值噪声比
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
#归一化图像数据（8位图像数据格式）
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# 位置编码（对应论文或文档的5.1章节）
class Embedder:
    def __init__(self, **kwargs):
        # 将传入的关键字参数保存到实例属性self.kwargs中
        self.kwargs = kwargs
        # 调用create_embedding_fn方法来创建嵌入函数相关的配置
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        # 用于存放一系列嵌入函数的列表，初始为空
        embed_fns = []
        # 获取输入维度，从传入的关键字参数（self.kwargs）中提取'input_dims'对应的值
        d = self.kwargs['input_dims']
        # 输出维度，初始值设为0，后续会根据嵌入函数的情况累加计算得到最终值
        out_dim = 0
        # 判断是否要包含原始输入作为嵌入的一部分（从关键字参数中获取'include_input'对应的值来判断）
        if self.kwargs['include_input']:
            # 如果包含原始输入，就添加一个简单的恒等函数（即直接返回输入本身）到嵌入函数列表中
            embed_fns.append(lambda x : x)
            # 相应地增加输出维度，增加的值为原始输入的维度d
            out_dim += d

        # 获取最大频率对应的以2为底的对数（从关键字参数中获取'max_freq_log2'对应的值）    
        max_freq = self.kwargs['max_freq_log2']
        # 获取频率的数量（从关键字参数中获取'num_freqs'对应的值）
        N_freqs = self.kwargs['num_freqs']
        
        # 根据log_sampling参数来确定频率带的生成方式，如果为True，则采用对数采样方式生成频率带，也就是以指数形式（以2为底）的线性间隔来生成频率值
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        # 如果log_sampling为False，则采用普通的线性间隔生成频率带（同样以2为底的指数形式）
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        # 遍历生成的每个频率值，针对每个频率都要结合周期性函数来构建嵌入函数
        for freq in freq_bands:
            # 遍历在关键字参数中指定的每个周期性函数p_fn，周期性函数用于对输入基于不同频率进行周期性的变换操作，以生成不同的嵌入表示特征
            for p_fn in self.kwargs['periodic_fns']:
                # 向嵌入函数列表中添加一个新的嵌入函数，该函数会先将输入乘以当前频率，然后应用对应的周期性函数进行处理，最终生成嵌入表示的一部分
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                # 每添加这样一个基于频率和周期性函数的嵌入函数，就相应地增加输出维度，增加的值为输入维度d，因为每个这样的函数都会为最终的嵌入表示贡献一定维度的数据
                out_dim += d
                    
        # 将生成的嵌入函数列表保存到实例属性self.embed_fns中，方便后续在embed方法里调用这些函数进行实际的嵌入操作
        self.embed_fns = embed_fns
        # 将最终确定的输出维度保存到实例属性self.out_dim中，后续在嵌入操作完成后，返回的嵌入表示结果的维度就是这个值
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # 使用torch.cat函数将每个嵌入函数应用到输入inputs后得到的结果按最后一维进行拼接，最终返回拼接后的结果作为完整的嵌入表示
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    #参数说明：
    #multires： 用于确定频率相关的参数，比如最大频率对应的以2为底的对数、频率的数量等设置会依赖它的值。
    #i：        控制返回哪种嵌入方式的标志，默认值为0，-1 时有特殊返回情况，其他值按常规配置构建嵌入器。

    #返回一个 `nn.Identity` 对象（这是一个恒等映射，输入什么就输出什么），以及维度值3。   
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    """
    构建一个字典 `embed_kwargs`，用于传递给 `Embedder` 类的初始化函数，来配置嵌入器的相关参数。
    - 'include_input': 设置为 `True`，表示嵌入结果中会包含原始输入作为一部分。
    - 'input_dims': 设定输入维度为3，说明输入的数据维度是3维的（例如可能是空间坐标等维度为3的数据）。
    -'max_freq_log2': 根据传入的 `multires` 值减1来确定最大频率对应的以2为底的对数，用于后续生成频率带的相关配置。
    - 'num_freqs': 直接使用 `multires` 的值作为频率的数量，即要生成多少个不同频率的嵌入函数。
    - 'log_sampling': 设置为 `True`，意味着采用对数采样方式来生成频率带（以指数形式（以2为底）的线性间隔生成频率值）。
    - 'periodic_fns': 提供了两个周期性函数 `torch.sin` 和 `torch.cos`，后续会基于不同频率结合这些函数构建嵌入函数，使得嵌入表示具有周期性特征。
    """

    #使用 `embed_kwargs` 字典作为关键字参数来实例化 `Embedder` 类，创建一个具体的嵌入器对象 `embedder_obj`，这个对象内部会根据传入的参数配置构建相应的嵌入函数等相关设置。
    embedder_obj = Embedder(**embed_kwargs)

    #创建一个匿名函数（lambda函数） `embed`，它接收输入 `x`，并利用之前创建的嵌入器对象 `eo`（默认绑定为 `embedder_obj`）调用其 `embed` 方法对输入 `x` 进行嵌入处理，返回嵌入后的结果
    embed = lambda x, eo=embedder_obj : eo.embed(x)

    """
    最终返回创建好的嵌入函数 `embed` 以及嵌入器对象 `embedder_obj` 的输出维度 `embedder_obj.out_dim`，
    输出维度信息可以用于后续操作中对输出结果维度的预期和处理等情况。
    """ 
    return embed, embedder_obj.out_dim




class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
       #用于构建NeRF模型的网络结构，设置各层的参数以及相关配置。

       #参数说明：
       #- D：整数类型，代表网络中全连接层的深度（层数），默认值为8，用于确定位置编码相关的线性层数量。
       #- W：整数类型，代表每层全连接层的神经元数量（宽度），默认值为256，控制每层网络的复杂度。
       #- input_ch：整数类型，输入的位置信息的通道数，默认值为3，例如可以表示三维空间坐标（x, y, z）等情况。
       #- input_ch_views：整数类型，输入的视角方向信息的通道数，默认值为3，比如可以表示视角方向的向量信息等。
       #- output_ch：整数类型，模型最终输出的通道数，默认值为4，例如可能包含颜色（RGB）以及体密度（alpha）等信息。
       #- skips：列表类型，其中的元素是整数，表示在哪些层进行跳跃连接（skip connection），默认包含元素4，用于融合浅层和深层的特征。
       #- use_viewdirs：布尔类型，用于判断是否使用视角方向信息来影响模型输出，如果为True，则在计算输出时会结合视角方向进行处理，否则不使用。
       
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        """
        创建一个名为 `pts_linears` 的ModuleList，用于存放处理位置信息的全连接层。
        首先添加第一层，它将输入的 `input_ch` 维度（位置信息维度）转换为 `W` 维度。
        然后通过列表推导式构建后续的 `D - 1` 层，对于不在 `skips` 列表中的层，是普通的从 `W` 维度到 `W` 维度的全连接层；
        而对于在 `skips` 列表中的层，会将当前层的输出与原始输入的 `input_ch` 维度进行拼接后再通过全连接层转换为 `W` 维度，实现跳跃连接。
        """        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        """
        这里有两种实现方式，一种是按照官方代码发布的方式，创建一个只包含一层的 `views_linears` ModuleList，
        这一层将视角方向信息（`input_ch_views` 维度）与前面位置编码部分输出的 `W` 维度信息拼接后转换为 `W//2` 维度。

        另一种是按照论文原本的描述实现方式（当前被注释掉了），除了第一层的转换外，还会继续添加 `D//2` 层从 `W//2` 维度到 `W//2` 维度的全连接层，
        用于进一步处理结合视角方向后的特征信息。
        """
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        


        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)

            #创建三个线性层：
            #- `feature_linear`：将位置编码部分最终输出的 `W` 维度特征进一步转换为 `W` 维度，用于后续操作。
            #- `alpha_linear`：将 `W` 维度特征转换为1维，通常用于得到体密度（alpha）信息。
            #- `rgb_linear`：将结合视角方向后处理得到的 `W//2` 维度特征转换为3维，用于得到颜色（RGB）信息。
        else:
            self.output_linear = nn.Linear(W, output_ch)
            #不使用视角方向信息，只创建一个 `output_linear` 线性层，
            #直接将位置编码部分最终输出的 `W` 维度特征转换为 `output_ch` 维度，得到模型最终输出。

    def forward(self, x):

        #将输入 `x` 按照维度在最后一维上进行分割，得到位置信息 `input_pts`（维度为 `self.input_ch`）和视角方向信息 `input_views`（维度为 `self.input_ch_views`）。
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
                #对位置编码部分的全连接层进行循环处理：
                #- 首先通过当前层 `self.pts_linears[i]` 对特征 `h` 进行线性变换。
                #- 然后应用ReLU激活函数进行非线性激活。
                #- 如果当前层索引 `i` 在 `skips` 列表中，则将原始的位置输入 `input_pts` 与当前层输出 `h` 在最后一维进行拼接，实现跳跃连接，融合浅层和深层特征。

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            """
            如果使用视角方向信息（`self.use_viewdirs` 为True）：
            - 首先通过 `alpha_linear` 层得到体密度（alpha）信息。
            - 再通过 `feature_linear` 层对位置编码部分的最终输出 `h` 进行特征转换。
            - 然后将转换后的特征 `feature` 与视角方向信息 `input_views` 在最后一维进行拼接，用于后续结合视角方向的特征处理。
            """
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
                """
                对结合视角方向后的全连接层进行循环处理：
                - 通过当前层 `self.views_linears[i]` 对特征 `h` 进行线性变换。
                - 接着应用ReLU激活函数进行非线性激活。
                """

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
            """
            - 通过 `rgb_linear` 层将处理后的特征转换为颜色（RGB）信息。
            - 最后将颜色信息 `rgb` 和体密度信息 `alpha` 在最后一维进行拼接，得到模型最终的输出 `outputs`，包含了颜色和体密度信息。
            """

        else:
            outputs = self.output_linear(h)
            """
            如果不使用视角方向信息（`self.use_viewdirs` 为False），直接通过 `output_linear` 层将位置编码部分最终输出的 `h` 转换为模型最终输出 `outputs`。
            """

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        """
        这个方法用于从Keras模型的权重加载到当前PyTorch的 `NeRF` 模型中，前提是 `use_viewdirs` 为True，
        因为如果不使用视角方向信息，当前加载权重的实现方式没有处理这种情况（代码中直接断言了这种限制）。
        """
        
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
            """
            循环加载处理位置信息的全连接层 `pts_linears` 的权重：
            - 根据索引 `i` 确定在传入的权重列表 `weights` 中的对应位置（权重和偏置是交替存储的，所以索引是 `2 * i` 和 `2 * i + 1`）。
            - 将权重数据从Numpy数组转换为PyTorch的Tensor，并赋值给当前层的 `weight.data` 属性；同理将偏置数据进行转换并赋值给 `bias.data` 属性。
            """
        
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))
        """
        加载 `feature_linear` 层的权重和偏置，同样是从传入的权重列表 `weights` 中根据对应的索引位置获取数据，
        然后转换并赋值给该层的 `weight.data` 和 `bias.data` 属性。
        """

        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))
        """
        加载 `views_linears` 中第一层（按照当前采用的官方代码实现方式只有一层）的权重和偏置，从权重列表中获取对应数据并进行转换赋值操作。
        """

        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))
        """
        加载 `rgb_linear` 层的权重和偏置，从权重列表中获取对应数据，转换后赋值给该层的 `weight` 和 `bias` 属性。
        """

        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
        """
        加载 `alpha_linear` 层的权重和偏置，按照对应索引从权重列表中获取数据，然后进行转换赋值操作，完成整个模型从Keras权重的加载过程。
        """

def get_rays(H, W, K, c2w):
    """
    根据图像的高度（H）、宽度（W）、相机内参矩阵（K）以及相机到世界坐标系的变换矩阵（c2w）来获取光线的起点和方向。

    参数:
    - H：图像的高度，整数类型，表示图像在垂直方向上的像素数量。
    - W：图像的宽度，整数类型，表示图像在水平方向上的像素数量。
    - K：相机内参矩阵，形状通常为(3, 3)的二维张量，包含了诸如焦距等相机内部参数信息。
    - c2w：相机到世界坐标系的变换矩阵，形状通常为(4, 4)的二维张量，用于将相机坐标系下的坐标转换到世界坐标系下。

    返回值:
    - rays_o：光线的起点，形状为(H, W, 3)的三维张量，表示在世界坐标系下每条光线起始位置的坐标。
    - rays_d：光线的方向，形状为(H, W, 3)的三维张量，表示在世界坐标系下每条光线的方向向量。
    """
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # PyTorch的meshgrid函数的索引方式为'ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # 将光线方向从相机坐标系旋转到世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 点积运算，等同于 [c2w.dot(dir) for dir in dirs]
    # 将相机坐标系的原点平移到世界坐标系，它就是所有光线的原点
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    """
    此函数功能与`get_rays`类似，但使用的是NumPy数组进行操作，根据图像的高度（H）、宽度（W）、相机内参矩阵（K）以及相机到世界坐标系的变换矩阵（c2w）来获取光线的起点和方向。

    参数:
    - H：图像的高度，整数类型，表示图像在垂直方向上的像素数量。
    - W：图像的宽度，整数类型，表示图像在水平方向上的像素数量。
    - K：相机内参矩阵，形状通常为(3, 3)的二维NumPy数组，包含了诸如焦距等相机内部参数信息。
    - c2w：相机到世界坐标系的变换矩阵，形状通常为(4, 4)的二维NumPy数组，用于将相机坐标系下的坐标转换到世界坐标系下。

    返回值:
    - rays_o：光线的起点，形状为(H, W, 3)的三维NumPy数组，表示在世界坐标系下每条光线起始位置的坐标。
    - rays_d：光线的方向，形状为(H, W, 3)的三维NumPy数组，表示在世界坐标系下每条光线的方向向量。
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # 将光线方向从相机坐标系旋转到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 点积运算，等同于 [c2w.dot(dir) for dir in dirs]
    # 将相机坐标系的原点平移到世界坐标系，它就是所有光线的原点
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    将光线转换到归一化设备坐标（Normalized Device Coordinates，NDC）空间。

    参数:
    - H：图像的高度，整数类型，表示图像在垂直方向上的像素数量。
    - W：图像的宽度，整数类型，表示图像在水平方向上的像素数量。
    - focal：相机焦距，浮点数类型，用于投影相关的计算。
    - near：近平面距离，浮点数类型，代表在场景中距离相机较近的平面位置。
    - rays_o：光线的起点，形状为(..., 3)的张量，在世界坐标系下每条光线起始位置的坐标，这里的“...”表示可以是任意维度前缀（例如可能是批量维度等）。
    - rays_d：光线的方向，形状为(..., 3)的张量，在世界坐标系下每条光线的方向向量，同样“...”表示可以是任意维度前缀。

    返回值:
    - rays_o：转换到NDC空间后的光线起点，形状与输入的rays_o一致，维度为(..., 3)的张量。
    - rays_d：转换到NDC空间后的光线方向，形状与输入的rays_d一致，维度为(..., 3)的张量。
    """
    # 将光线起点平移到近平面
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    # 投影操作
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    
    return rays_o, rays_d


# 分层采样（对应论文5.2章节）
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    根据给定的离散区间（bins）和对应的权重（weights）进行概率密度函数（Probability Density Function，PDF）采样，可选择确定性采样或随机采样，并支持测试模式下用固定的随机数进行采样。

    参数:
    - bins：离散区间，形状为(batch, len(bins))的二维张量，代表了采样的区间边界信息。
    - weights：权重，形状为(batch, len(bins))的二维张量，对应每个区间的权重值，用于构建概率密度函数。
    - N_samples：采样数量，整数类型，指定要采集的样本数量。
    - det：布尔类型，若为True，则进行确定性采样（例如均匀采样）；若为False，则进行随机采样，默认值为False。
    - pytest：布尔类型，若为True，则在测试模式下会用NumPy的固定随机数覆盖默认的随机采样值，默认值为False。

    返回值:
    - samples：采样得到的样本，形状为(batch, N_samples)的二维张量，是根据给定的权重和区间进行采样后得到的数据。
    """
    # 获取概率密度函数（PDF）
    weights = weights + 1e-5  # 防止出现NaN值
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # 进行均匀采样
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # 测试模式下，用NumPy的固定随机数覆盖u的值
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 求逆累积分布函数（CDF）
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
