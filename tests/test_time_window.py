# -*- coding: utf-8 -*-
"""parse_time_window 的回归与修复测试（2026-11-03）。

背景：线上实测发现两处**误识别**，比漏识别更危险（漏=这轮没加权；错=加权到错误时段）：
  1. 「上上个月」被子串匹配成「上个月」→ 返回 (30,59)，应为 (60,89)
  2. 「最近一个月」被静态词「最近」抢先匹配成 (0,6)，应为 (0,29)

纪律：新增词表规则必须同时有「正例」和「守护例」——守护例钉住旧行为不被新规则带偏。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_retriever import parse_time_window  # noqa: E402


class TestMisparseFixes(unittest.TestCase):
    """两处误识别的修复（本次改动目标）。"""

    def test_last_last_month_not_parsed_as_last_month(self):
        self.assertEqual(parse_time_window("上上个月在忙什么"), (60, 89),
                         "「上上个月」不得被子串「上个月」吃掉")

    def test_last_last_month_variants(self):
        for q in ("上上个月", "上上月", "上上个月底做的事"):
            self.assertEqual(parse_time_window(q), (60, 89), f"{q} 应返回两个月前窗口")

    def test_recent_n_months(self):
        self.assertEqual(parse_time_window("最近3个月的流水"), (0, 89),
                         "「最近 3 个月」应走动态规则，不被静态「最近」(0,6) 抢先")

    def test_recent_n_months_chinese_numeral(self):
        self.assertEqual(parse_time_window("最近一个月"), (0, 29))
        self.assertEqual(parse_time_window("最近三个月"), (0, 89))

    def test_recent_n_weeks_and_days(self):
        self.assertEqual(parse_time_window("最近两周"), (0, 13))
        self.assertEqual(parse_time_window("最近10天"), (0, 9))


class TestGuards(unittest.TestCase):
    """守护例：修复不得改变这些既有行为。"""

    def test_last_month_unchanged(self):
        self.assertEqual(parse_time_window("上个月在忙什么"), (30, 59))

    def test_bare_recent_unchanged(self):
        self.assertEqual(parse_time_window("最近在忙什么"), (0, 6))

    def test_recent_variants_unchanged(self):
        self.assertEqual(parse_time_window("这几天怎么样"), (0, 6))
        self.assertEqual(parse_time_window("这阵子折腾的"), (0, 6))

    def test_past_n_unchanged(self):
        self.assertEqual(parse_time_window("过去30天有哪些事件"), (0, 29))
        self.assertEqual(parse_time_window("过去两周"), (0, 13))

    def test_week_and_day_words_unchanged(self):
        self.assertEqual(parse_time_window("上周部署了什么"), (7, 13))
        self.assertEqual(parse_time_window("今天做了什么"), (0, 0))
        self.assertEqual(parse_time_window("昨天那个报错"), (1, 1))
        self.assertEqual(parse_time_window("三天前那次故障"), (3, 3))

    def test_no_time_word_returns_none(self):
        for q in ("我喜欢喝什么", "9 月份我搞了哪些网络的事", "我家服务器在哪"):
            self.assertIsNone(parse_time_window(q), f"{q} 目前不应被解析（绝对月份未支持）")


if __name__ == "__main__":
    unittest.main()
