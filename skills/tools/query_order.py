from skills.registry import register_skill
from skills._mock import mock_order


@register_skill
class QueryOrderSkill:
    name = "query_order"
    description = (
        "查询玩家投注注单的结算状态、派彩金额、赛事信息。"
        "当用户询问 \"某订单结算了吗\" / \"我那笔注单怎么样\" / \"我的投注派彩了没\" 时调用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "订单号/注单号，通常以 OD/BT/RD 等字母开头，后跟 8-12 位数字",
            },
            "user_id": {
                "type": "string",
                "description": "玩家账号（可选，若未提供则忽略）",
            },
        },
        "required": ["order_id"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        order_id = (args.get("order_id") or "").strip()
        if not order_id:
            return {"ok": False, "error": "缺少必填参数 order_id"}
        return mock_order(order_id)
