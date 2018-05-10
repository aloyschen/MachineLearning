import os
import colorsys
import random
import keras.backend as K
import tensorflow as tf
from keras.models import load_model

class YOLO:
    def __init__(self, obj_threshold, nms_threshold):
        """
        构造函数
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值
            nms_threshold: nms阈值
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = './model_data/coco_classes.txt'
        self.class_names = self._get_class()
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def save_model_for_production(self, model_path, export_path):
        """
        读取训练好的模型，导出为Tensorflow serving 支持的模型格式
        """
        K.set_learning_phase(0)
        model = load_model(model_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        model_input = tf.saved_model.utils.build_tensor_info(model.input)
        model_boxes = tf.saved_model.utils.build_tensor_info(model.output[0])
        model_scores = tf.saved_model.utils.build_tensor_info(model.output[1])
        model_classes = tf.saved_model.utils.build_tensor_info(model.output[2])

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs' : model_input},
                outputs={'boxes' : model_boxes,
                         'classes' : model_classes,
                         'scores' : model_scores},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(
                sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict': prediction_signature,
                })
            builder.save()


    def yolo_eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """

        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]
        boxes = []
        box_scores = []
        input_shape = K.shape(yolo_outputs[0])[1 : 3] * 32
        for i in range(len(yolo_outputs)):
            _anchors = [anchors[index] for index in anchor_mask[i]]
            _boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[i], _anchors, len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = K.concatenate(boxes, axis = 0)
        box_scores = K.concatenate(box_scores, axis = 0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis = 0)
        scores_ = K.concatenate(scores_, axis = 0)
        classes_ = K.concatenate(classes_, axis = 0)
        return boxes_, scores_, classes_


    def yolo_boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """

        :param feats:
        :param anchors:
        :param class_num:
        :param input_shape:
        :param image_shape:
        :return:
        """
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, classes_num, input_shape)
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores


    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes



    def yolo_head(self, feats, anchors, num_classes, input_shape):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (box_xy + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

        return box_xy, box_wh, box_confidence, box_class_probs