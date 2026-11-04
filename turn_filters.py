# -*- coding: utf-8 -*-
"""turn_filters — 入库前的轮次级过滤（纯 stdlib，无 Hermes / httpx 依赖）。

为什么单独一个模块：`__init__.py` 依赖 agent 运行时，测试要么造 stub 要么绕；
把纯函数放这里，测试用 importlib 直接加载就能裸跑。

## 纯确认词轮次不入库（2026-11-04）

服务端对 blob 只有「memo 为空才不产事件」一道闸，**没有信息量门槛**。人造样本
实测：5 轮纯确认词（好/继续/嗯嗯可以/那就这么办吧/好的好的）flush 后照样产 1 条
事件（「用户表达了同意和确认的态度」「用户对当前对话内容表示认可」）**并且进画像**。

收词原则（宁可漏，不乱杀）：
- 只跳「整句由确认词拼成」的轮次（`好的好的` / `嗯嗯` / `OK`）；
- `好，那就这么办吧` / `继续之前那个方案` / `好的但别动 nginx` 一律照常入库 —— 带决策；
- 去掉标点后长度上限 12，再长不可能是纯确认。
"""
from __future__ import annotations

import re

# 允许重复拼装（「好的好的」「嗯嗯嗯」），所以按词素集合做贪心切分，不做整句匹配。
ACK_TOKENS = (
    "好的", "好滴", "好呀", "好呢", "好", "嗯嗯", "嗯呢", "嗯",
    "可以", "行", "对的", "对呀", "对", "是的", "是",
    "继续", "收到", "明白", "没问题", "谢谢", "辛苦", "不错", "赞",
    "okay", "ok", "yes",
    # 语气词/助词：允许「可以呀」「行吧」「辛苦了」这类纯语气收尾
    "呀", "呢", "吧", "啊", "哦", "嘛", "啦", "了", "的",
)

_MAX_ACK_CHARS = 12
_NON_WORD = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _NON_WORD.sub("", text or "").lower()


def is_ack_only_turn(text: str) -> bool:
    """整句是否只是确认词（无任何决策/对象/信息）。空串返回 False。"""
    t = _normalize(text)
    if not t or len(t) > _MAX_ACK_CHARS:
        return False
    tokens = sorted(ACK_TOKENS, key=len, reverse=True)
    while t:
        for tok in tokens:
            if t.startswith(tok):
                t = t[len(tok):]
                break
        else:
            return False
    return True
