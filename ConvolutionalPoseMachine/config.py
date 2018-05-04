# 配置文件
# 训练数据关节点名称
joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
# 热度图高斯平方差
sigma = 1.0
# tfrecord文件路径
train_record = './data/train.tfrecord'
test_record = './data/test.tfrecord'
# 随机旋转图片次数
repeat_num = 5
# 存储图像的路径
img_dir = '/Users/gaochen3/sina_study/machineLearning/ConvolutionalPoseMachine/dataset/mpii/image'
# 标注数据文件
train_data_file = '/Users/gaochen3/sina_study/machineLearning/ConvolutionalPoseMachine/dataset/mpii/dataset.txt'
# 输入图片的大小
in_size = 300
# 处理数据的进程数
process_num = 10
# 验证集比例
valid_rate = 0.2
# 是否保存热度图
save_heatMap = True
# 是否训练
is_train = True
# 是否保存图片
is_save_img = True
# 图片使用的颜色空间
color = 'RGB'