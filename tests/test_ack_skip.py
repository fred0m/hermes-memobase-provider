# -*- coding: utf-8 -*-
"""纯确认词轮次不入库（ack-only turn skip）—— 2026-11-04。

为什么：服务端对 blob 只有「memo 为空才不产事件」一道闸，没有信息量门槛。
人造样本实测：5 轮纯确认词（好/继续/嗯嗯可以/那就这么办吧/好的好的）flush 后
照样产 1 条事件（「用户表达了同意和确认的态度」「用户对当前对话内容表示认可」）
**并且进画像**。真实频率低（8200 条 user 消息里纯确认词 1.5%），但成批出现就污染。

收词原则（宁可漏，不乱杀）：
- 只跳「整句由确认词拼成」的轮次（「好的好的」「嗯嗯」「OK」），
- 「好，那就这么办吧」「继续之前那个方案」「好的但别动 nginx」一律照常入库 —— 它们带决策。
- 长度上限 12（去掉标点后），再长就不可能是纯确认。
"""
import importlib.util
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 直接按文件加载：turn_filters 是纯 stdlib 模块，不需要 agent 运行时 stub
_spec = importlib.util.spec_from_file_location(
    "memobase_turn_filters", os.path.join(REPO_ROOT, "turn_filters.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

is_ack_only_turn = _mod.is_ack_only_turn


class TestAckOnlyPositive(unittest.TestCase):
    def test_plain_acks(self):
        for t in ("好", "好的", "嗯", "嗯嗯", "继续", "可以", "行", "收到", "明白",
                  "OK", "okay", "谢谢", "辛苦了", "没问题", "对", "是的"):
            self.assertTrue(is_ack_only_turn(t), f"应判纯确认: {t!r}")

    def test_acks_with_punctuation_and_repeat(self):
        for t in ("好。", "好的！", "嗯嗯～", "好的好的", "嗯嗯嗯", "可以呀", "行吧", " OK! "):
            self.assertTrue(is_ack_only_turn(t), f"应判纯确认: {t!r}")


class TestAckOnlyNegative(unittest.TestCase):
    def test_decision_carrying_short_turns_kept(self):
        """反例守护：带决策/对象的短句一律不跳，这是本规则的风险面。"""
        for t in (
            "好，那就这么办吧",
            "继续之前那个方案",
            "好的但别动 nginx",
            "行，先观察一周",
            "嗯，我看看日志",
            "可以，明天再说",
        ):
            self.assertFalse(is_ack_only_turn(t), f"带决策不该跳: {t!r}")

    def test_normal_content_kept(self):
        for t in ("这个报错怎么弄", "9 月份我搞了哪些网络的事", "继续把 memobase 的 G1 改成收敛重写"):
            self.assertFalse(is_ack_only_turn(t))

    def test_empty_kept(self):
        # 空串由 sync_turn 前面的既有判断处理，这里不该判成 ack（避免两处语义打架）
        self.assertFalse(is_ack_only_turn(""))
        self.assertFalse(is_ack_only_turn("   "))


if __name__ == "__main__":
    unittest.main()
