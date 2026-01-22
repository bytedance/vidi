"""
Copyright 2025 Intelligent Editing Team.
"""
import datetime
import logging
import logging.handlers
import os
import sys
import torch
import requests
import torch.nn.functional as F
import math

from vidi.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def space_to_depth(x, m_size=2):
    """
    将输入的(B, C, H, W) tensor，按空间块(m_size, m_size)重排到channel维。
    Args:
        x: 输入Tensor, shape为 (B, C, H, W)
        m_size: 空间块大小（如2表示2x2 patch）
    Returns:
        输出Tensor, shape为 (B, C * m_size * m_size, H // m_size, W // m_size)
    """
    B, C, H, W = x.shape
    assert H % m_size == 0 and W % m_size == 0, "H和W必须能被m_size整除"
    # 1. 先reshape，把空间拆成m_size x m_size小块
    x = x.reshape(B, C, H // m_size, m_size, W // m_size, m_size)
    # 2. 调整维度，把小块合并到channel维
    x = x.permute(0, 1, 3, 5, 2, 4)  # (B, C, m_size, m_size, H//m_size, W//m_size)
    x = x.reshape(B, C * m_size * m_size, H // m_size, W // m_size)
    return x

def resize_by_tokens(x, max_tokens):
    """
    x: 输入Tensor, shape (B, C, H, W)
    max_tokens: B*H*W不超过的最大token数
    mode: F.interpolate的插值模式
    返回: 缩放后的x
    """
    x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
    B, C, H, W = x.shape
   
    # B不变，只缩H和W
    ratio = math.sqrt(max_tokens / (B * H * W))  # H和W都按这个比例缩放
    
    temp_H = int(H * ratio)
    temp_W = int(W * ratio)

    new_H = max(10, temp_H - temp_H%2)
    new_W = max(10, temp_W - temp_W%2)

    return new_H, new_W 
