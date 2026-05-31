#!/usr/bin/env python3
import json
from pathlib import Path


def build_titles():
    return {
        "bet_abnormal": [
            "注单未找到先核验账号与注单号",
            "注单状态处理中说明与等待时效",
            "开赛前误投申请取消窗口内",
            "开赛后取消申请驳回解释",
            "重复下注争议核查流程",
            "注单号与账号不匹配处理",
            "多笔同场下注合并查询",
            "注单已结算但余额未变",
            "注单作废原因标准口径",
            "赛事延期触发作废说明",
            "赛事腰斩后注单处理",
            "封盘边界时间争议",
            "盘口切换导致注单疑问",
            "系统判定无效注单申诉",
            "部分成功部分失败下单异常",
            "跨端显示不一致核查",
            "注单异常需后台复核",
            "超时未回单升级催办",
            "夜间值班场景延迟告知",
            "高情绪投诉下注单安抚闭环",
        ],
        "parlay_settlement": [
            "二串一基础结算规则说明",
            "多串过关计算示例",
            "串关中一场作废如何重算",
            "串关含走盘处理说明",
            "串关含提前结算说明",
            "串关含中断赛处理",
            "不同联赛规则差异告知",
            "串关派奖时间解释",
            "赔率变动是否影响已下注",
            "串关最小最大投注限制",
            "串关是否支持取消统一口径",
            "串关待结算但单关已完赛",
            "串关部分场次推迟影响",
            "串关与活动叠加规则",
            "串关派奖金额不一致解释",
            "串关含特殊玩法说明",
            "串关结果争议证据收集",
            "串关结算失败自动重试",
            "串关长期未结算升级流程",
            "串关规则FAQ标准回复",
        ],
        "live_delay": [
            "滚球注单延迟结算原因说明",
            "VAR回看导致延迟模板",
            "数据源同步慢导致延迟",
            "盘口频繁波动下延迟解释",
            "红牌点球关键事件延迟",
            "中场节间结算节奏说明",
            "比赛暂停导致滚单挂起",
            "加时补时触发结算延后",
            "滚球早结后复核调整",
            "赔率冻结期间状态解释",
            "单场多笔滚单分批结算",
            "未结算先安抚后承诺时效",
            "跨时区结算时间解释",
            "延迟超阈值升级模板",
            "滚球异常需后台复核",
            "比分修正导致重算说明",
            "取消作废与延迟区别",
            "高频追问好了没节奏回复",
            "夜间赛事批量延迟公告",
            "滚球延迟结算FAQ模板",
        ],
        "odds_anomaly": [
            "已下注后赔率变化不追溯",
            "赔率明显异常触发风控复核",
            "盘口临场切换说明",
            "封盘前后边界下单争议",
            "盘口误开导致作废口径",
            "多数据源赔率冲突处理",
            "同赛事实时赔率跳变解释",
            "截图赔率与系统不一致",
            "串关单腿赔率修正说明",
            "赔率异常引发结算重算",
            "盘口关闭后恢复告知",
            "盘口不可用时引导替代玩法",
            "赔率异常投诉工单模板",
            "赔率争议证据清单",
            "大额单触发赔率保护机制",
            "早盘与临场赔率差异解释",
            "重大事件后盘口重开说明",
            "市场暂停非系统故障口径",
            "赔率盘口FAQ短问短答",
            "异常赔率争辩的合规回复",
        ],
        "risk_limit": [
            "单注限额触发提示",
            "单日累计限额触发说明",
            "赛事级限红提示",
            "盘口级限红提示",
            "账户风控审核中说明",
            "身份核验补件引导",
            "异地登录触发风控",
            "频繁改密改绑触发限制",
            "可疑资金触发限制口径",
            "申诉入口与工单回执",
            "申诉进度查询模板",
            "材料不足退回补充模板",
            "限红解除成功通知",
            "限红解除失败解释",
            "高情绪为什么限我安抚",
            "合规拒绝赌博诱导话术",
            "合规拒绝代付跑分请求",
            "合规拒绝伪造资料请求",
            "未成年人暗示终止服务",
            "风控FAQ标准答复",
        ],
    }


def build_meta():
    return {
        "bet_abnormal": {
            "intent": "注单异常查询",
            "sub_intent": "注单查询/取消作废",
            "target_count": 150,
        },
        "parlay_settlement": {
            "intent": "结算/派奖",
            "sub_intent": "串关规则与结算",
            "target_count": 125,
        },
        "live_delay": {
            "intent": "结算/派奖",
            "sub_intent": "滚球延迟结算",
            "target_count": 100,
        },
        "odds_anomaly": {
            "intent": "下注/赔率/赛程",
            "sub_intent": "异常赔率/盘口变更",
            "target_count": 100,
        },
        "risk_limit": {
            "intent": "账户异常/风控",
            "sub_intent": "限红风控提示与申诉",
            "target_count": 125,
        },
    }


def build_template_row(scenario, idx, title, meta):
    return {
        "template_id": f"{scenario[:2].upper()}{idx:02d}",
        "scenario": scenario,
        "intent": meta["intent"],
        "sub_intent": meta["sub_intent"],
        "title": title,
        "target_count": meta["target_count"],
        "priority": 5 if idx <= 5 else 3,
        "required_slots": ["<ACCOUNT>", "<BET_ID>", "<STATUS>", "<NEXT_ACTION>"],
        "optional_slots": [
            "<ORDER_ID>",
            "<SPORT>",
            "<LEAGUE>",
            "<MATCH>",
            "<MARKET>",
            "<ODDS>",
            "<STAKE>",
            "<PAYOUT>",
            "<VOID_REASON>",
            "<DELAY_REASON>",
            "<RISK_REASON>",
            "<LIMIT_TYPE>",
            "<LIMIT_VALUE>",
            "<SLA_MIN>",
            "<SLA_HOUR>",
            "<TICKET_ID>",
            "<APPEAL_PATH>",
        ],
        "dialog_flow": ["identify", "verify", "explain", "action", "sla", "close"],
        "allowed_tones": [
            "professional_neutral",
            "empathetic",
            "concise_actionable",
            "firm_compliance",
            "high_pressure_deescalation",
        ],
        "turn_range": "4-10",
    }


def main():
    root = Path("/home/ubuntu/qwen35a3b_finetune/templates")
    root.mkdir(parents=True, exist_ok=True)
    out = root / "templates_100.jsonl"
    titles = build_titles()
    meta = build_meta()

    rows = []
    for scenario, title_list in titles.items():
        for i, title in enumerate(title_list, start=1):
            rows.append(build_template_row(scenario, i, title, meta[scenario]))

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} templates -> {out}")


if __name__ == "__main__":
    main()
