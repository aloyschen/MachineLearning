#
#单机多卡(GPU)
#
##################################################
TensorFlow_Submit \
--appName=tensorflow_cifar_distribution   \
--archives=/user/chaochao1/gpu_lib/Python.zip#Python \
--files=./cifar10.py,./cifar10_eval.py,./cifar10_input.py,./cifar10_multi_gpu_train.py  \
--worker_memory=4096 \
--ps_memory=4096 \
--num_ps=10  \
--num_worker=10  \
--worker_cores=1 \
--worker_gpu_cores=1 \
--job_node_label_expression=gpu-p100 \
--data_dir=hdfs://ns1-backup/user/chaochao1/cifar10_data \
--train_dir=hdfs://ns1-backup/user/chaochao1/cifar10_multi_gpu_model_out/test_$RANDOM \
--command=Python/bin/python cifar10_multi_gpu_train.py test=100 test2=200\
