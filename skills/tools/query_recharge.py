from skills.registry import register_skill
from skills._mock import mock_recharge


@register_skill
class QueryRechargeSkill:
    name = "query_recharge"
    description = (
        "查询玩家的充值/入款记录的处理状态、渠道、到账时间。"
        "当用户询问 \"我充值还没到账\" / \"上次入款怎么样了\" / \"充值订单查一下\" 时调用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_no": {"type": "string", "description": "充值订单号（与 user_id 至少一项）"},
            "user_id": {"type": "string", "description": "玩家账号（与 order_no 至少一项）"},
        },
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        order_no = (args.get("order_no") or "").strip() or None
        user_id = (args.get("user_id") or "").strip() or None
        if not order_no and not user_id:
            return {"ok": False, "error": "至少需要提供 order_no 或 user_id 之一"}
        return mock_recharge(order_no, user_id)
