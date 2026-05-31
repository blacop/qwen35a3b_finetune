"""Mock 后台数据。
等运维拿到真实 API 文档后，把这里替换为真实 HTTP 调用即可（推荐建一个 _http.py 处理鉴权/超时/限频）。
"""
import random
import hashlib


def _seed_for(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def mock_order(order_id: str) -> dict:
    rnd = random.Random(_seed_for(order_id))
    statuses = ["未结算", "已结算-赢", "已结算-输", "已结算-平局", "已取消"]
    return {
        "ok": True,
        "order_id": order_id,
        "status": rnd.choice(statuses),
        "bet_amount": rnd.choice([50, 100, 200, 500, 1000]),
        "payout": rnd.choice([0, 95, 180, 380, 750]),
        "match_name": rnd.choice([
            "皇家马德里 vs 巴塞罗那",
            "曼联 vs 利物浦",
            "湖人 vs 勇士",
            "广东宏远 vs 辽宁本钢",
        ]),
        "settled_at": "2026-04-25 18:30:00" if rnd.random() > 0.3 else None,
        "_mock": True,
    }


def mock_recharge(order_no=None, user_id=None) -> dict:
    seed = order_no or user_id or "default"
    rnd = random.Random(_seed_for(seed))
    statuses = ["成功", "处理中", "失败-超时", "失败-渠道异常", "审核中"]
    return {
        "ok": True,
        "order_no": order_no or f"RC{rnd.randint(10000000, 99999999)}",
        "user_id": user_id or "user_demo",
        "amount": rnd.choice([100, 200, 500, 1000, 5000]),
        "channel": rnd.choice(["银行卡-工商", "USDT-TRC20", "支付宝", "微信"]),
        "status": rnd.choice(statuses),
        "submit_time": "2026-04-25 14:20:00",
        "complete_time": "2026-04-25 14:35:00" if rnd.random() > 0.3 else None,
        "_mock": True,
    }


def mock_member(user_id: str) -> dict:
    rnd = random.Random(_seed_for(user_id))
    return {
        "ok": True,
        "user_id": user_id,
        "balance": round(rnd.uniform(0, 50000), 2),
        "vip_level": rnd.choice(["VIP1", "VIP2", "VIP3", "VIP4", "VIP5"]),
        "register_at": "2025-08-15 10:00:00",
        "total_deposit": round(rnd.uniform(1000, 200000), 2),
        "total_turnover": round(rnd.uniform(5000, 1000000), 2),
        "risk_tag": rnd.choice(["normal", "normal", "normal", "watch", "frozen"]),
        "_mock": True,
    }
