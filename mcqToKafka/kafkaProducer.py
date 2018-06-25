import memcache
import logging
import logging.handlers
import multiprocessing
import mcqToKafka.config as config
from kafka import KafkaProducer
import argparse



def listener_config():
    """
    Introduction
    ------------
        想在多进程中把所有日志输入到一份Log中，需要有一个进程一直监听接受log，写入日志中
    """
    root = logging.getLogger()
    # 可以管理日志文件大小，当日志超过大小创建一个新的文件写入日志
    h = logging.handlers.RotatingFileHandler('mcq.log', mode = 'a', maxBytes = 10240, backupCount = 3)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)


def worker_config(queue):
    """
    Introduction
    ------------
        worker的log设置
    Parameters
    ----------
        queue: log事件写入队列
    """
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)


def listener_process(queue, config):
    """
    Introduction
    ------------
        通过队列接受日志信息，写入到日志文件中
    Parameters
    ----------
        queue: 接受日志的队列
        configurer: 设置日志格式
    """
    config()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file = sys.stderr)
            traceback.print_exc(file = sys.stderr)


class McqToKafka:
    def __init__(self, McqSever, KafkaSevers, McqTopic, KafkaTopic, process_num):
        """
        Introduction
        ------------
            读取Mcq数据写入kafka，构造函数
        Parameters
        ----------
            Mcqsevers: Mcq客户端域名
            Mcqtopic: Mcq队列的读key
            KafkaTopic: kafka的topic
            kafkaSevers: kafka的域名
        """
        self.McqSever = McqSever
        self.KafkaSevers = KafkaSevers
        self.McqTopic = McqTopic
        self.KafkaTopic = KafkaTopic
        self.process_num = process_num


    def worker_process(self, queue, config, mcqServer):
        """
        Introduction
        ------------
            从Mcq队列中读取图片信息写入到Kafka中
        Parameters
        ----------
            queue: 日志队列
            config: worker日志设置
            mcqServer: mcq的server地址
        """
        config(queue)
        mcq = memcache.Client(mcqServer)
        kafkaProducer = KafkaProducer(bootstrap_servers = self.KafkaSevers)
        try:
            while True:
                image = mcq.get(self.McqTopic)
                if image != None:
                    result = kafkaProducer.send(self.KafkaTopic, image)
                    logging.info(result.get())
        except Exception as e:
            logging.error(e)


    def multiproces(self):
        """
        Introduction
        ------------
            一个域名对应多个读取进程
        """
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target = listener_process, args = (queue, listener_config))
        listener.start()
        workers = []
        for _ in range(self.process_num):
            worker = multiprocessing.Process(target = self.worker_process,args = (queue, worker_config, self.McqSever))
            workers.append(worker)
            worker.start()
        for w in workers:
            w.join()
        queue.put_nowait(None)
        listener.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read mcq and write kafka')
    parser.add_argument('McqServers', metavar='N', type=str, nargs='+',
                        help='Mcq Severs (tc or yf)')
    args = parser.parse_args()
    if args.McqServers == 'tc':
        McqToKafka = McqToKafka(config.McqServerstc, config.KafkaServers, config.McqKey, config.KafkaTopic, config.process_num)
        McqToKafka.multiproces()
    if args.McqServers == 'yf':
        McqToKafka = McqToKafka(config.McqServerstc, config.KafkaServers, config.McqKey, config.KafkaTopic, config.process_num)
        McqToKafka.multiproces()