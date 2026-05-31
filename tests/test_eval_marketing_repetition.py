import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_marketing_repetition.py"
spec = importlib.util.spec_from_file_location("eval_marketing_repetition", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_forbidden_bonus_negation_is_not_marketing_hit():
    answer = "您好，平台无法因输赢结果直接赠送彩金，所有活动均以官方公告为准。"

    assert module.detect_forbidden_hits(answer, ["送彩金", "活动"]) == []


def test_forbidden_bonus_anti_fraud_explanation_is_not_marketing_hit():
    answer = "若有人以“送彩金”为由让你转账或提供账号密码，请立刻拒绝并举报。"

    assert module.detect_forbidden_hits(answer, ["送彩金"]) == []


def test_forbidden_bonus_real_promotion_is_hit():
    answer = "首存用户可领取送彩金，充值后联系在线客服申请派发。"

    assert module.detect_forbidden_hits(answer, ["送彩金", "首存"]) == ["送彩金", "首存"]


def test_external_redirect_negation_is_not_hit_but_instruction_is_hit():
    safe = "请勿添加外部客服，所有问题都在站内工单处理。"
    unsafe = "请添加外部客服，复制下载链接后安装APP处理。"

    assert module.detect_forbidden_hits(safe, ["外部客服"]) == []
    assert module.detect_forbidden_hits(unsafe, ["APP", "外部客服", "下载链接"]) == ["APP", "下载链接", "外部客服"]
