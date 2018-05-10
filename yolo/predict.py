import grpc
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto
from YoloDetect import YOLO
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from PIL import ImageFont, ImageDraw
from tensorflow_serving.apis import prediction_service_pb2

def process_image(img):
    """Resize, reduce and expand image.
    # Argument:
        img: original image.
    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(416) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = 416 - new_size[1]
    delta_h = 416 - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [128, 128, 128]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    image = np.array(new_im, dtype='float32')
    print(image.shape)
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype(int))
        left = max(0, np.floor(left + 0.5).astype(int))
        right = min(image.shape[1], np.floor(right + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype(int))

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 4)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate top, left, bottom, right: {} {} {} {}'.format(top, left, bottom, right))
    cv2.imshow('image', image)
    cv2.waitKey()


def detect_image(image_file, obj_threshold, box_threshold, host, port = 9000):
    """
    Introduction
    ------------
        请求yoloV3模型server，根据返回结果输出图片中物体位置和类别
    Parameters
    ----------
        image_file: 图片路径
        obj_threshold: 检测为物体的阈值
        box_threshold: 物体检测框的阈值
        class_file: 所有类别的名字
        host: serve的ip地址
        port: serve的端口号
    """
    start = time.time()
    outs = []
    yolo = YOLO(obj_threshold, box_threshold)
    yolo.save_model_for_production('./model_data/yolo.h5', './prod_models')
    image = cv2.imread(image_file)
    pimage = process_image(image)
    # 建立连接
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'Yolo'
    request.model_spec.signature_name = 'predict'
    request.inputs['inputs'].CopyFrom(make_tensor_proto(pimage))

    try:
        response = stub.Predict.future(request, 40)
        for key in response.result().outputs:
            tensor_proto = response.result().outputs[key]
            outs.append(tf.contrib.util.make_ndarray(tensor_proto))
    except grpc.RpcError as error:
        print(error)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2, ))
    yolo_out1 = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 255))
    yolo_out2 = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 255))
    yolo_out3 =  tf.placeholder(dtype = tf.float32, shape = (None, None, None, 255))
    outs_placeholder = [yolo_out1, yolo_out2, yolo_out3]
    boxes, scores, classes = yolo.yolo_eval(outs_placeholder, input_image_shape)
    image_shape = np.array([image.shape[1], image.shape[0]]).reshape((2,))
    with tf.Session() as sess:
        out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={yolo_out1 : outs[0], yolo_out2 : outs[1], yolo_out3 : outs[2], input_image_shape : image_shape})

    end = time.time()

    print('time: {0:.2f}s'.format(end - start))
    if out_boxes is not None:
        draw(image, out_boxes, out_scores, out_classes, yolo.class_names)

    return image

if __name__ == "__main__":
    detect_image('./test.jpg', 0.3, 0.7, '13.112.30.246')