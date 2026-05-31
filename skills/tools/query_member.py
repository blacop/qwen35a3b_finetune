from skills.registry import register_skill
from skills._mock import mock_member


@register_skill
class QueryMemberSkill:
    name = "query_member"
    description = (
        "查询玩家账户的余额、VIP 等级、累计存款/流水、风控状态。"
        "当用户询问 \"我余额多少\" / \"我现在 VIP 几级\" / \"我账户怎么样\" 时调用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "玩家账号（必填）",
            },
        },
        "required": ["user_id"],
    }

    @staticmethod
    async def execute(args: dict) -> dict:
        user_id = (args.get("user_id") or "").strip()
        if not user_id:
            return {"ok": False, "error": "缺少必填参数 user_id"}
        return mock_member(user_id)
