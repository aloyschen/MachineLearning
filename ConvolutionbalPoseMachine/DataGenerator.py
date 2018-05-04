import os
import cv2
import config
import random
import numpy as np
from skimage import transform
import scipy.misc as scm

class DataGenerator(object):
    """
    该类是用于产生训练集、测试集和验证集
    Inputs shape:
        number of Image * (height : 256) * (weight : 256) * (channels : 3)
    Outputs shape:
        number of Image * (number of stacks) * (height : self.out_size) * (weight : self.out_size) * (OutputDimendion : 16)
    """
    def __init__(self, img_dir, train_data_file, in_size = 368, out_size = None):
        """
        构造函数
        Parameters
        ----------
            img_dir: 图片路径
            train_data_file: 训练标签文件
            in_size: 输入图片大小
            out_size: 输出图片大小
        """
        self.joints_list = config.joints_list
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.in_size = in_size
        if out_size is None:
            self.out_size = self.in_size / 8
        else:
            self.out_size = out_size
        # 用来区分不同人体的字母范围
        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # 读取所有图片路径到内存中
        self.images = os.listdir(self.img_dir)

    def _create_train_data(self):
        """
        该函数是用来将标注数据解析，然后存储成一个字典
        格式如下：
            key: img_name
            value: {'box', 'joints', 'weights'}
        """
        # 训练数据
        self.train_data = []
        # 没有关节点标注的数据
        self.no_label = []
        # 图片和标注对应的字典
        self.data_dict = {}
        with open(self.train_data_file, 'r') as file:
            for line in file.readlines():
                line = line.strip().split(' ')
                img_name = line[0]
                box = [int(line[i]) for i in range(1, 5)]
                joints = [int(line[i]) for i in range(5, len(line))]
                # 判断标注数据中是否有关节点的位置
                if joints == [-1] * len(joints):
                    self.no_label.append(img_name)
                    continue
                else:
                    joints = np.reshape(joints, [-1, 2])
                    # weights 记录每个关节点是否有坐标标注，如果没有则将权重置为0
                    weights = [1] * joints.shape[0]
                    for index in range(joints.shape[0]):
                        if joints[index][0] == -1 and joints[index][1] == -1:
                            weights[index] = 0
                self.train_data.append(img_name)
                self.data_dict[img_name] = {'box' : box, 'Joints' : joints, 'weights' : weights}

        # 将训练集数据打乱
        random.shuffle(self.train_data)


    def _is_complete_sample(self, name):
        """
        判断一个样本中关节点数据是否缺失
        Parameters
        ----------
            name: 图片的名字
        Returns
        -------
            False: 有缺失
            True: 没有缺失
        """
        data = self.data_dict.get(name)
        if data is None:
            return False
        for em in data['joints']:
            if em[0] == -1 and em[1] == -1:
                return False
        return True


    def _create_train_valid_sets(self, validation_rate = 0.1):
        """
        该函数是用来切分训练集和验证集
        Parameters
        ----------
            validation_rate: 验证集占比
        """
        train_data_length = len(self.train_data)
        val_length = int(train_data_length * validation_rate)
        self.train_set = self.train_data[ : train_data_length - val_length]
        pre_set = self.train_data[train_data_length - val_length : ]
        self.val_set = []
        for element in pre_set:
            # 如果数据集关节点坐标完整则加入测试集
            if self._is_complete_sample(element):
                self.val_set.append(element)
            else:
                self.train_set.append(element)



    def _makeGaussian(self, heatMap_size, sigma = 3, center = None):
        """
        产生一个二维高斯分布函数, x和y相同的方形区域, 中心为(heatMap_size / 2, heatMap_size / 2), 标准差为sigma
        Parameters
        ----------
            heatMap_size: 高斯分布大小
            sigma: 高斯分布的标准差
            center: 高斯分布的中心
        Returns
        -------
            二维高斯分布
        """
        x = np.arange(0, heatMap_size, dtype = np.float32)
        y = np.arange(0, heatMap_size, dtype = np.float32)[:, np.newaxis]

        if center is None:
            center = (heatMap_size // 2, heatMap_size // 2)
        return np.exp(-4 * np.log(2) * ((x - center[0]) ** 2 + (y - center[1]) ** 2) / (sigma ** 2))


    def _generate_heatMap(self, heatMap_size, joints, weight):
        """
        在图片中关节点的位置生成热度图
        Parameters
        ----------
            heatMap_size: 热度图大小
            joints: 关节点坐标
            weight: 关节点权重值
        Returns
        -------
            关节点热度图
        """
        joints_num = joints.shape[0]
        heatMap = np.zeros([heatMap_size, heatMap_size, joints_num], dtype = np.float32)
        for index in range(joints_num):
            if weight[index] == 1:
                heatMap[:, :, index] = self._makeGaussian(heatMap_size, config.sigma, (joints[index][0], joints[index][1]))
            else:
                heatMap[:, :, index] = np.zeros([heatMap_size, heatMap_size])
        return heatMap


    def open_image(self, img_name, color = 'RGB'):
        """
        读取图片
        Parameters
        ----------
            color: 读取的颜色格式
        Returns
        -------
            image: 读取的图片
        """
        if img_name[-1] in self.letter:
            img_name = img_name[: -1]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def _crop_data(self, height, width, box, joints, boxp = 0.05):
        """
        获取包含人体矩形框的中心点坐标，并为下一步提取图片上正方形矩形框做padding
        Parameters
        ----------
            height: 图像的高度
            width: 图像的宽度
            box: 人体矩形框左上角和右下角的坐标
            joints: 关节点坐标
            boxp: 需要扩大box的比例
        Returns
        -------
            padding: 图像二维方向填充的矩阵
            crop_box: 包含人体矩形框的中心坐标和长宽值
        """
        joints_copy = joints.copy()
        padding = [[0, 0], [0, 0], [0, 0]]
        # 如果人体框的左上角坐标缺失，则用关节点坐标填充
        if box[0] == -1 and box[1] == -1:
            joints_copy[joints == -1] = 1e5
            box[0] = np.min(joints_copy[:, 0])
            box[1] = np.min(joints_copy[:, 1])
        box_weight = box[2] - box[0] + 1
        box_height = box[3] - box[1] + 1
        box[0] = int(box[0] - boxp * box_weight)
        box[1] = int(box[1] - boxp * box_height)
        box[2] = int(box[2] + boxp * box_weight)
        box[3] = int(box[3] + boxp * box_height)

        # 看是否越界
        if box[0] < 0:
            box[0] = 0
        if box[1] < 0:
            box[1] = 0
        if box[2] > width -1:
            box[2] = width -1
        if box[3] > height -1:
            box[3] = height -1

        # 计算新的box长宽
        box_weight_new = box[2] - box[0]
        box_height_new = box[3] - box[1]

        # 计算返回的矩形框中心坐标和长宽
        crop_box = [box[0] + box_weight_new // 2, box[1] + box_height_new // 2, box_weight_new, box_height_new]
        # 需要提取正方形区域的图片，因此如果box在图像的边缘，需要做padding处理
        if box_height_new > box_weight_new:
            bounds = (crop_box[0] - box_height_new // 2, crop_box[0] + box_height_new // 2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width -1:
                padding[1][1] = abs(width - bounds[1])
        elif box_weight_new > box_height_new:
            bounds = (crop_box[1] - box_weight_new // 2, crop_box[1] + box_weight_new // 2)
            if bounds[0] < 0:
                padding[0][0] = abs(bounds[0])
            if bounds[1] > height - 1:
                padding[0][1] = abs(height - bounds[1])
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        return padding, crop_box


    def _relative_joints(self, crop_box, padding, joints, toSize):
        """
        该函数是将关节点坐标转换为热度图中的相对坐标
        Parameters
        ----------
            crop_box: 包含人体的矩形框
            padding: 为了获取正方形区域的填充
            joints: 关节点坐标
            toSize: 热度图大小
        Returns
        -------
            relative_joints: 相对关节点坐标
        """
        relative_joints = joints.copy()
        relative_joints += [padding[1][0], padding[0][0]]
        max_length = max(crop_box[2], crop_box[3])
        relative_joints = [crop_box[0] - max_length //2, crop_box[1] - max_length // 2]
        relative_joints = relative_joints * toSize / (max_length + 0.0000001)
        return relative_joints


    def _augment(self, img, heatMap, joints, weights, ):
    def get_sample(self, img_name, queue, color = 'RGB'):
        if img_name is not None:
            joints = self.data_dict[img_name]['joints']
            box = self.data_dict[img_name]['box']
            weights = self.data_dict[img_name]['weights']
            img = self.open_image(img_name, color)
            padd, crop_box = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
            relative_joints = self._relative_joints(crop_box, padd, joints,toSize=self.out_size)
            heatMap = self._generate_heatMap(self.out_size, relative_joints, weights)
            img = np.pad(img, padd, mode = 'constans')
            max_length = max(crop_box[2], crop_box[3])
            img = img[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2, crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
            img = scm.imresize(img, (self.in_size, self.in_size))
