# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import functools
from datetime import datetime


class TimeRecoder:
    __slots__ = ["_start_time", "_has_start"]

    def __init__(self):
        self._start_time = datetime.now()
        self._has_start = False

    def start(self, msg=""):
        self._start_time = datetime.now()
        self._has_start = True
        if msg:
            print(f"Start: {msg}")

    def now_and_reset(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now_and_reset`!")
        self._has_start = False
        end_time = datetime.now()
        print(f"End: {pre_msg} {end_time - self._start_time}")
        self.start()

    def now(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now`!")
        self._has_start = False
        end_time = datetime.now()
        print(f"[{end_time}] {pre_msg} {end_time - self._start_time}")

    @staticmethod
    def decorator(start_msg="", end_pre_msg=""):
        """as a decorator"""
        _temp_obj = TimeRecoder()

        def _timer(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _temp_obj.start(start_msg)
                results = func(*args, **kwargs)
                _temp_obj.now(end_pre_msg)
                return results

            return wrapper

        return _timer
