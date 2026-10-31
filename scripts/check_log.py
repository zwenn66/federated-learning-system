# -*- coding: utf-8 -*-
import os
import re

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.log")
d = open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""
rs = re.findall(r"\[ROUND (\d+)\]", d)
print("最后轮次:", rs[-1] if rs else "无")
print("FitMetrics 数:", d.count("FitMetrics"))
print("EvaluateMetrics 数:", d.count("EvaluateMetrics"))
# 统计每个轮次
import collections
if rs:
    cnt = collections.Counter(rs)
    print("轮次分布:", dict(sorted(cnt.items())))
