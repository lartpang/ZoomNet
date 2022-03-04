# -*- coding: utf-8 -*-
# @Time    : 2021/3/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import argparse
import os.path
import subprocess
import time
from enum import Enum
from multiprocessing import Process

import pynvml

pynvml.nvmlInit()


class STATUS(Enum):
    NORMAL = 0
    GPU_BUSY = 1


class MyProcess:
    slot_idx = -1
    curr_task_id = 0

    def __init__(
        self,
        interpreter_path,
        gpu_id,
        verbose=True,
        stdin=None,
        stdout=None,
        stderr=None,
        num_cmds=None,
        max_used_ratio=0.5,
    ):
        super().__init__()
        self.gpu_id = gpu_id
        self.interpreter_path = interpreter_path
        self.verbose = verbose
        self.num_cmds = num_cmds

        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

        self.sub_proc = None
        self.proc = None

        self.gpu_handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.max_used_ratio = max_used_ratio
        MyProcess.slot_idx += 1

    def __str__(self):
        return f"[ID {self.slot_idx} INFO] NEW PROCESS SLOT ON GPU {self.gpu_id} IS CREATED!"

    def _used_ratio(self, used, total):
        return used / total

    def get_used_mem(self, return_ratio=False):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handler)
        if return_ratio:
            return self._used_ratio(meminfo.used, meminfo.total)
        return meminfo.used

    def _create_sub_proc(self, cmd=""):
        self.sub_proc = subprocess.Popen(
            args=f"CUDA_VISIBLE_DEVICES={self.gpu_id} {self.interpreter_path} -u {cmd}",
            stdin=self.stdin,
            stdout=self.stdout,
            stderr=self.stderr,
            shell=True,
            executable="bash",
            env=None,
            close_fds=True,
            bufsize=1,
            text=True,
            encoding="utf-8",
        )
        print(f"[NEW TASK PID: {self.sub_proc.pid}] {self.sub_proc.args}")

        if self.verbose:
            if self.stdout is not None and self.sub_proc is not None:
                for l in self.sub_proc.stdout:
                    print(f"[ID: {self.curr_task_id}/{self.num_cmds} GPU: {self.gpu_id}] {l}", end="")

    def create_and_start_proc(self, cmd=None):
        if (used_mem := self.get_used_mem(return_ratio=True)) > self.max_used_ratio:
            # TODO: 当前的判定方式并不是太准确。最好的方式是由程序提供设置周期数的选项(`--num-epochs`)，
            #   首先按照num_epoch=1来进行初步的运行，并统计各个命令对应使用的显存。
            #   之后根据这些程序实际使用的显存来安排后续的操作。
            #   这可能需要程序对输出可以实现覆盖式(`--overwrite`)操作。
            self.status = STATUS.GPU_BUSY
            print(
                f"[ID {self.slot_idx} WARN] the memory usage of the GPU {self.gpu_id} is currently {used_mem}, "
                f"which exceeds the maximum threshold {self.max_used_ratio}."
            )
            return

        print(f"[ID {self.slot_idx} INFO] {cmd}")
        MyProcess.curr_task_id += 1
        self.proc = Process(target=self._create_sub_proc, kwargs=dict(cmd=cmd))
        self.proc.start()

        # 只有成功创建并启动了进城后才改变状态
        self.status = STATUS.NORMAL

    def is_alive(self):
        if self.status == STATUS.NORMAL:
            return self.proc.is_alive()
        return False


def read_cmds_from_txt(path):
    with open(path, encoding="utf-8", mode="r") as f:
        cmds = []
        for line in f:
            line = line.rstrip()
            if line and line[0].isalpha():
                cmds.append(line)
    return cmds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpreter", type=str, required=True, help="The path of your interpreter you want to use.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print the output of the subprocess.")
    parser.add_argument(
        "--gpu-pool", nargs="+", type=int, default=[0], help="The pool containing all ids of your gpu devices."
    )
    parser.add_argument("--max-workers", type=int, help="The max number of the workers.")
    parser.add_argument(
        "--cmd-pool",
        type=str,
        required=True,
        help="The text file containing all your commands. It will be combined with `interpreter`.",
    )
    parser.add_argument("--poll-interval", type=int, default=0, help="The interval of the poll.")
    parser.add_argument("--max-used-ratio", type=float, default=0.5)
    args = parser.parse_args()
    args.interpreter = os.path.abspath(args.interpreter)
    if args.max_workers is None:
        args.max_workers = len(args.gpu_pool)
    return args


def main():
    args = get_args()
    print("[YOUR CONFIG]\n" + str(args))
    cmd_pool = read_cmds_from_txt(path=args.cmd_pool)
    print("[YOUR CMDS]\n" + "\n".join(cmd_pool))

    num_gpus = len(args.gpu_pool)

    print("[CREATE PROCESS OBJECTS]")
    proc_slots = []
    for i in range(min(args.max_workers, len(cmd_pool))):  # 确保slots数量小于等于命令数量
        gpu_id = i % num_gpus
        proc = MyProcess(
            interpreter_path=args.interpreter,
            gpu_id=args.gpu_pool[gpu_id],
            verbose=args.verbose,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            num_cmds=len(cmd_pool),
            max_used_ratio=args.max_used_ratio,
        )
        print(proc)
        proc_slots.append(proc)

    cmd_pool.reverse()  # 后面的操作是按照栈的形式处理的，所以这里翻转一下
    for p in proc_slots:
        if len(cmd_pool) == 0:  # 确保出栈不会异常
            break
        cmd = cmd_pool.pop()  # 指令出栈
        p.create_and_start_proc(cmd=cmd)
        if p.status == STATUS.GPU_BUSY:  # 当前GPU显存不足，暂先跳过
            cmd_pool.append(cmd)  # 指令未能顺利执行，重新入栈
            continue

    is_normal_ending = True
    while proc_slots:
        # the pool of the processes is not empty
        for slot_idx, p in enumerate(proc_slots):  # polling
            if not p.is_alive():
                if len(cmd_pool) == 0:  # 指令均在执行或者已被执行
                    del proc_slots[slot_idx]
                    print("[NO MORE COMMANDS, DELETE THE PROCESS SLOT!]")
                    break

                cmd = cmd_pool.pop()
                p.create_and_start_proc(cmd=cmd)
                if p.status == STATUS.GPU_BUSY:  # 当前GPU显存不足，暂先跳过
                    cmd_pool.append(cmd)  # 指令未能顺利执行，重新入栈
                    continue

        if proc_slots and all([_p.status == STATUS.GPU_BUSY for _p in proc_slots]):
            # 所有GPU都被外部程序占用，直接退出。因为如果我们的程序正常执行时，状态是NORMAL
            if args.poll_interval > 0:
                print(f"[ALL GPUS ARE BUSY, WAITING {args.poll_interval} SECONDS!]")
                time.sleep(args.poll_interval)
            else:
                print("[ALL GPUS ARE BUSY, EXIT THE LOOP!]")
                proc_slots.clear()
                is_normal_ending = False
                break

        time.sleep(1)

    if is_normal_ending:
        print("[ALL COMMANDS HAVE BEEN COMPLETED!]")


if __name__ == "__main__":
    main()
