import logging
import sys

def getLogger(filename, show_level=logging.INFO):

    # 创建日志的实例
    logger = logging.getLogger("main")

    # 指定Logger的输出格式
    formatter = logging.Formatter(
        fmt="%(asctime)s-%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # 创建日志：
    # 文件日志
    file_handler = logging.FileHandler(filename=filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 终端日志
    consle_handler = logging.StreamHandler(sys.stdout)
    consle_handler.setFormatter(formatter)

    # 设置默认的日志级别
    logger.setLevel(show_level)

    # 把文件日志和终端日志添加到日志处理器中
    logger.addHandler(file_handler)
    logger.addHandler(consle_handler)

    return logger
