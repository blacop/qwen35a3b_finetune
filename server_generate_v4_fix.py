#!/usr/bin/env python3
import json, random, uuid, sys, os
from collections import Counter

random.seed(2026)

QUOTA = {"赛事变更": 2500, "升级场景强化": 2000, "必含要素强化": 1500}

TEAMS = [
    ("曼城","利物浦"),("巴萨","皇马"),("拜仁","多特"),("国米","AC米兰"),("阿森纳","切尔西"),
    ("曼联","热刺"),("那不勒斯","尤文"),("巴黎","里昂"),("上海海港","北京国安"),("广州队","山东泰山"),
    ("湖人","凯尔特人"),("勇士","雄鹿"),("掘金","热火"),("阿根廷","巴西"),("法国","德国"),
    ("浦和红钻","鹿岛鹿角"),("全北现代","蔚山现代"),("弗拉门戈","帕尔梅拉斯"),
    ("独行侠","太阳"),("76人","尼克斯"),("辽宁","广东"),("新疆","浙江"),
]
LEAGUES = ["英超","西甲","德甲","意甲","法甲","欧冠","中超","NBA","CBA","世预赛","欧洲杯","亚冠","欧联","J联赛","K联赛"]
BETS = ["独赢","让球","大小","半全场","波胆","角球大小","篮球让分","篮球大小分","总进球","进球单双"]
PARLAYS = ["2串1","3串1","4串1","5串1","3串4"]
AMTS = ["50","100","200","500","1000","2000","5000","10000","20000","50000"]
ODDS = ["1.52","1.75","1.88","1.95","2.10","2.65","3.50","5.50","8.00"]
SLAS = ["3-5分钟内","15分钟内","30分钟内","1小时内","2小时内","24小时内"]
EMPATHY = ["非常理解您的心情，","很抱歉给您带来不便，","感谢您的耐心等待，","理解您的着急，","非常抱歉出现这个情况，","完全理解您的担忧，"]
CLOSING = ["请问还有其他需要帮助的吗？","如有其他问题随时联系我们。","祝您投注顺利！","感谢您的理解与配合！","如需进一步协助，随时找我。"]
DEVICES = ["iOS APP","安卓APP","H5网页","PC客户端"]
MINUTES = ["第35分钟","第48分钟","第62分钟","第78分钟","第88分钟","上半场","下半场","加时赛","伤停补时"]
CANCEL_REASONS = ["场地问题","恶劣天气","安全事件","球队弃权","球员罢赛","自然灾害","突发公共事件","场馆停电"]
POSTPONE_REASONS = ["暴雨天气","场地维护","球队航班延误","疫情防控","安全隐患排查","转播方要求调整","多队赛程冲突"]
CHANGE_TYPES = ["改期","取消","中断后未恢复","腰斩","延期","推迟","中止","暂停后官方判定无效"]

def R(pool): return random.choice(pool)
def acc(): return f"user{random.randint(10000,99999)}"
def oid(): return f"OD{random.randint(10**11,10**12-1)}"
def match():
    h,a = R(TEAMS); return f"{h}vs{a}"
def rid(): return f"REC_{uuid.uuid4().hex[:12]}"

def match_change_1():
    m = match()
    return [
        ("user", f"{m}比赛改期了，我的注单{oid()}怎么算？"),
        ("assistant", f"{R(EMPATHY)}赛事改期的情况我帮您核实。请提供您的账号和注单号，我查看该注单的当前状态和结算规则。"),
        ("user", f"账号{acc()}，注单号刚才说了，{R(POSTPONE_REASONS)}导致改期的"),
        ("assistant", f"已查到您的注单。根据平台赛事变更规则：如比赛在原定开赛时间起48小时内重新开赛，注单保持有效，按最终赛果结算；如超过48小时或官方取消，注单按作废处理，本金全额退还至您的账户余额。当前该赛事状态为「延期待定」，建议您关注后续赛程公告。{R(CLOSING)}"),
    ], False

def match_change_2():
    m = match()
    return [
        ("user", f"{m}比赛取消了，我下了{R(AMTS)}，钱什么时候退？"),
        ("assistant", f"{R(EMPATHY)}赛事取消的注单会按规则处理退款。请提供您的账号和注单号，我立即帮您核实退款进度。"),
        ("user", f"账号{acc()}，注单{oid()}，你们快点退"),
        ("assistant", f"已核实，该赛事因{R(CANCEL_REASONS)}被官方取消。您的注单本金将全额退还。系统正在批量处理涉及该赛事的所有注单退款，预计{R(SLAS)}到账。如超时未收到，我会帮您提交至运营后台加急跟进。"),
        ("user", "太慢了，能快点吗？"),
        ("assistant", f"理解您的着急。我已将您的账号和注单号提交至运营专员做加急处理，并标注了优先退款。有进展会第一时间通知您。{R(CLOSING)}"),
    ], True

def match_change_3():
    m = match()
    return [
        ("user", f"{m}比赛打到{R(MINUTES)}中断了，我买的{R(BETS)}怎么结算？"),
        ("assistant", f"{R(EMPATHY)}赛事中断的结算需等官方最终裁定。请提供您的账号和注单号，我帮您查看当前处理状态。"),
        ("user", f"账号{acc()}，注单{oid()}，中断前我是赢的"),
        ("assistant", f"已查到您的注单。赛事中断的处理规则如下：如官方裁定比赛有效（如已过特定时间节点），按中断时的比分结算；如裁定无效，注单作废退还本金。目前该赛事官方尚未出具最终裁定，系统暂时冻结该注单的结算。"),
        ("user", "那要等多久？中断前比分对我有利"),
        ("assistant", f"通常需等待官方24-48小时内的裁定公告。我已将您的注单号标记跟踪并提交至运营后台，裁定结果出来后会第一时间通知您并按规则结算。如您有赛事中断时的截图，建议保留以备申诉。{R(CLOSING)}"),
    ], True

def match_change_4():
    m = match()
    return [
        ("user", f"什么是赛事腰斩？{m}是腰斩了吗？我的注单会怎样？"),
        ("assistant", f"{R(EMPATHY)}腰斩是指比赛因不可抗力在正式结束前被终止，且未在规定时间内恢复的情况。请提供您的账号和注单号，我帮您查看该赛事的官方状态。"),
        ("user", f"账号{acc()}，注单{oid()}"),
        ("assistant", f"已查到。该赛事目前的官方状态为「{R(['腰斩','中断待裁定','延期'])}」。按平台规则：腰斩的比赛，已有明确结果的盘口正常结算，未产生明确结果的盘口按作废处理退还本金。您的{R(BETS)}注单属于{R(['已有明确结果，会正常结算','未产生明确结果，将按作废退还本金'])}。{R(CLOSING)}"),
    ], False

def match_change_5():
    m = match()
    return [
        ("user", f"我的{R(PARLAYS)}里有一场{m}改期了，整个串关怎么算？"),
        ("assistant", f"{R(EMPATHY)}串关中有场次改期会影响整单结算。请提供您的账号和注单号，我帮您确认具体影响。"),
        ("user", f"账号{acc()}，注单{oid()}"),
        ("assistant", f"已核实。串关中改期场次的处理：如该场在48小时内重新开赛，串关保持有效等待全部场次出结果；如超时或取消，该场按走水处理（赔率1.00），剩余有效场次赔率连乘计算派彩。您的注单中改期场次目前状态为「待定」，建议等待赛程更新。{R(CLOSING)}"),
    ], False

def match_change_6():
    m1, m2 = match(), match()
    return [
        ("user", f"今天好几场比赛{R(CHANGE_TYPES)}，{m1}和{m2}，我下了好几单怎么办？"),
        ("assistant", f"{R(EMPATHY)}多场赛事变更确实影响较大。请提供您的账号，我帮您批量查看所有受影响的注单。"),
        ("user", f"账号{acc()}，有{R(['3','4','5','6'])}个注单"),
        ("assistant", f"已查到您的账户下有多笔注单涉及赛事变更。由于涉及注单较多，我已将您的账号和所有相关注单号提交至运营专员做统一处理。运营同事会逐笔核实每张注单的赛事状态和对应的结算规则，预计{R(SLAS)}内给您完整反馈。"),
        ("user", "好的，辛苦了"),
        ("assistant", f"不客气。期间如有任何疑问随时联系我，我会持续跟进处理进度。{R(CLOSING)}"),
    ], True

def match_change_7():
    m = match()
    return [
        ("user", f"{m}延期了，我还没下注呢，新的开赛时间是什么时候？"),
        ("assistant", f"{R(EMPATHY)}赛事延期后新的开赛时间需等官方公布。目前该赛事在平台上的盘口已暂时关闭。"),
        ("user", "什么时候重新开盘？"),
        ("assistant", f"官方确认新开赛时间后，平台会同步更新赛程并重新开盘。您可以在赛事列表中关注该场比赛的状态更新，开盘后即可正常投注。{R(CLOSING)}"),
    ], False

def match_change_8():
    m = match()
    return [
        ("user", f"{m}比赛主客场互换了，我买的让球{R(ODDS)}还算数吗？"),
        ("assistant", f"{R(EMPATHY)}主客场互换属于赛事信息变更的一种。请提供您的账号和注单号，我帮您核实平台的处理规则。"),
        ("user", f"账号{acc()}，注单{oid()}"),
        ("assistant", f"已查到您的注单。对于主客场互换的情况：如果盘口是以指定主客场开出的（如主队让球），变更后该盘口按作废处理退还本金；如果是中性盘口（不区分主客），则不受影响正常结算。您的注单涉及让球盘，我已提交至运营后台核实具体处理方式，预计{R(SLAS)}内反馈。{R(CLOSING)}"),
    ], True

def match_change_9():
    m = match()
    return [
        ("user", f"上次{m}{R(CHANGE_TYPES)}，说退款结果一直没到账"),
        ("assistant", f"{R(EMPATHY)}退款延迟给您造成不便。请提供您的账号和注单号，我帮您催促查询退款进度。"),
        ("user", f"账号{acc()}，注单{oid()}，都等了{R(['两天','三天','一周'])}了"),
        ("assistant", f"已查到您的退款记录。系统显示该笔退款已于{R(['昨天','前天','今天早上'])}发起处理，但可能因{R(['系统结算队列拥堵','批量处理延迟','财务审核排队'])}尚未到账。我已将您的账号提交至运营专员做加急跟进，确保尽快完成退款到账。预计{R(SLAS)}处理完毕。"),
        ("user", "好的赶紧"),
        ("assistant", f"已加急标注。到账后您可在资金明细中查看。{R(CLOSING)}"),
    ], True

def match_change_10():
    return [
        ("user", f"你们赛事变更规则太复杂了，能不能简单说说？"),
        ("assistant", f"{R(EMPATHY)}我帮您梳理一下赛事变更的核心规则：\n\n1. 赛事改期：原开赛时间48小时内重新开赛则注单保持有效；超过48小时则注单作废退本金\n2. 赛事取消：注单作废，本金全额退还\n3. 赛事腰斩/中断：已有明确结果的盘口正常结算，未有结果的退本金\n4. 串关中有变更场次：变更场次按走水（赔率1.00）处理，不影响其他场次\n\n如果您有具体注单需要查询，请提供账号和注单号，我帮您逐笔确认。{R(CLOSING)}"),
    ], False

def esc_1():
    return [
        ("user", f"你们平台太垃圾了！我的{R(AMTS)}到现在没结算！再不处理我就投诉到{R(['监管部门','消费者协会','媒体'])}！"),
        ("assistant", f"{R(EMPATHY)}非常理解您的不满情绪。请提供您的账号和注单号，我立即帮您核实并升级处理。"),
        ("user", f"账号{acc()}，注单{oid()}，你们必须给我个说法"),
        ("assistant", f"已记录您的诉求。鉴于问题的紧急性，我已将您的账号、注单号和详细情况提交至运营主管直接跟进处理。运营专员会在{R(SLAS)}内主动联系您，给出明确的处理结论和方案。期间请保留好相关截图。"),
        ("user", "行，我等着"),
        ("assistant", f"一定会尽快给您满意的答复。{R(CLOSING)}"),
    ], True

def esc_2():
    return [
        ("user", f"这个问题我已经联系了{R(['两次','三次','好几次'])}客服了，每次都说在处理，一直没结果"),
        ("assistant", f"{R(EMPATHY)}多次联系仍未解决，给您造成了很大困扰。请提供您的账号和之前的工单号（如有），我直接帮您升级到运营主管处理。"),
        ("user", f"账号{acc()}，注单{oid()}，之前的工单号不记得了"),
        ("assistant", f"没关系，我已通过您的账号查到历史记录。确认此前已有{R(['2','3'])}次相关咨询记录但未最终解决。我已将此问题直接升级至运营主管，并附上完整的历史沟通记录，确保不再需要您重复说明情况。预计{R(SLAS)}内会有专人联系您。"),
        ("user", "这次一定要解决"),
        ("assistant", f"主管级别跟进会确保给出最终解决方案。如仍有问题可直接要求转主管处理。{R(CLOSING)}"),
    ], True

def esc_3():
    big_amt = R(["50000","80000","100000","200000"])
    return [
        ("user", f"我有一笔{big_amt}的{R(['提现','注单','充值'])}出了问题"),
        ("assistant", f"{R(EMPATHY)}大额问题我帮您重点跟进。请提供您的账号和相关单号。"),
        ("user", f"账号{acc()}，单号{oid()}"),
        ("assistant", f"已核实，该笔金额为{big_amt}元。由于金额较大，我已将您的账号和单号直接提交至运营专员做优先处理，同时通知财务团队同步核查。预计{R(SLAS)}内给您明确反馈。请保留好转账凭证或注单截图以备核对。"),
        ("user", "好的，一定要处理好"),
        ("assistant", f"大额问题会优先处理，请放心。{R(CLOSING)}"),
    ], True

def esc_4():
    return [
        ("user", f"你们{R(DEVICES)}刚才{R(['闪退','崩溃','卡死','白屏'])}，我{R(['下注失败扣了钱','操作到一半中断','投注被取消了'])}"),
        ("assistant", f"{R(EMPATHY)}系统故障影响您的操作，非常抱歉。请提供您的账号和故障发生的大概时间，我帮您核实是否有资金异常。"),
        ("user", f"账号{acc()}，大概{R(['5分钟前','10分钟前','半小时前'])}"),
        ("assistant", f"已记录。系统故障导致的投注异常需要技术团队核查操作日志。我已将您的账号和时间点提交至运营后台，技术团队会排查该时段的系统日志。如确认因系统故障导致的资金异常，会按原路径冲正。预计{R(SLAS)}反馈结果。"),
        ("user", "要是确认是你们系统问题呢？"),
        ("assistant", f"如确认为平台系统故障，会全额补偿受影响的资金。核查结果会通过站内信通知您。{R(CLOSING)}"),
    ], True

def esc_5():
    return [
        ("user", f"{match()}现在赔率多少？"),
        ("assistant", f"您可以在赛事详情页查看{R(BETS)}的实时赔率。赔率随市场实时变动，以页面显示为准。{R(CLOSING)}"),
    ], False

def esc_6():
    return [
        ("user", f"帮我查下账号{acc()}余额还有多少"),
        ("assistant", f"您可以登录后在「我的账户」页面查看实时余额和资金明细。如果登录遇到问题，请告知我具体报错信息。{R(CLOSING)}"),
    ], False

def esc_7():
    return [
        ("user", f"我提现{R(AMTS)}，{R(['24小时','两天','三天'])}了还没到账"),
        ("assistant", f"{R(EMPATHY)}提现长时间未到账确实让人担心。请提供您的账号和提现订单号，我帮您查询具体原因。"),
        ("user", f"账号{acc()}，提现单号{oid()}"),
        ("assistant", f"已查到该笔提现记录。当前状态显示为{R(['财务审核中','银行处理中','风控复核中'])}。由于已超出正常处理时限，我已将您的账号和订单号提交至运营专员做加急跟进，同时通知财务团队核实具体卡在哪个环节。预计{R(SLAS)}内给您明确答复。"),
        ("user", "行赶紧催催"),
        ("assistant", f"已加急催促。到账后请在资金明细中确认。{R(CLOSING)}"),
    ], True

def esc_8():
    return [
        ("user", f"{R(PARLAYS)}是什么意思？怎么玩？"),
        ("assistant", f"串关是将多场比赛的投注合并为一注，所有场次都猜中才算赢。赔率为各场赔率连乘，收益更高但难度也更大。如果其中一场走水，该场赔率按1.00计算，不影响其他场次。您可以在投注页面选择多场比赛后点击「串关」按钮操作。{R(CLOSING)}"),
    ], False

def esc_9():
    return [
        ("user", f"我刚充值{R(AMTS)}，用{R(['银行卡','USDT','支付宝','微信'])}转的，余额没变"),
        ("assistant", f"{R(EMPATHY)}充值未到账需要核实转账状态。请提供您的账号和转账凭证（时间、金额、渠道）。"),
        ("user", f"账号{acc()}，{R(['10分钟前','半小时前','1小时前'])}转的"),
        ("assistant", f"已记录。充值到账延迟可能因{R(['银行清算延迟','渠道对账时差','系统匹配队列'])}导致。我已将您的账号和充值信息提交至运营后台核查入账流水。如确认转账成功但未入账，会手动补入。预计{R(SLAS)}处理完毕。请保留好转账截图。"),
        ("user", "好的"),
        ("assistant", f"到账后会在资金明细中显示。{R(CLOSING)}"),
    ], True

def esc_10():
    return [
        ("user", f"{match()}最终比分是多少？"),
        ("assistant", f"您可以在赛事详情页或「已结束」列表中查看最终比分和详细数据。注单结算以平台数据源确认的官方赛果为准。{R(CLOSING)}"),
    ], False

MUST_INCLUDE_ELEMENTS = {
    "verify": ["请提供您的账号和注单号","请提供您的会员账号","请提供账号和相关订单号","请告知您的账号","请提供您的账号信息"],
    "action": ["已为您提交至运营专员处理","已提交至运营后台核实","已帮您生成加急工单","已将您的诉求升级至运营团队","已提交运营后台加急跟进","我已记录并提交后台处理"],
    "sla": [f"预计{s}内给您反馈" for s in SLAS] + [f"预计{s}内处理完毕" for s in SLAS],
    "record": ["请保留好注单截图以备核对","请保留好转账凭证","建议截图保留","请保留相关记录"],
}

def must_inc_1():
    m = match()
    return [
        ("user", f"我的注单{oid()}有问题，{m}的{R(BETS)}"),
        ("assistant", f"{R(EMPATHY)}{R(MUST_INCLUDE_ELEMENTS['verify'])}，我立即帮您核实注单状态。"),
        ("user", f"账号{acc()}"),
        ("assistant", f"已查到您的注单。当前该注单状态为{R(['待结算','已结算','异常冻结'])}。{R(MUST_INCLUDE_ELEMENTS['action'])}，{R(MUST_INCLUDE_ELEMENTS['sla'])}。{R(MUST_INCLUDE_ELEMENTS['record'])}。{R(CLOSING)}"),
    ], True

def must_inc_2():
    return [
        ("user", f"提款{R(AMTS)}一直没到账"),
        ("assistant", f"{R(EMPATHY)}{R(MUST_INCLUDE_ELEMENTS['verify'])}和提现订单号，我帮您查询提款进度。"),
        ("user", f"账号{acc()}，订单{oid()}"),
        ("assistant", f"已查到该笔提款记录，当前处于{R(['财务审核','银行处理','风控复核'])}阶段。{R(MUST_INCLUDE_ELEMENTS['action'])}，{R(MUST_INCLUDE_ELEMENTS['sla'])}。到账后您可在资金明细中确认。{R(MUST_INCLUDE_ELEMENTS['record'])}。{R(CLOSING)}"),
    ], True

def must_inc_3():
    return [
        ("user", f"赔率不对，我下注时明明是{R(ODDS)}"),
        ("assistant", f"{R(EMPATHY)}{R(MUST_INCLUDE_ELEMENTS['verify'])}，我帮您核对下注时锁定的赔率记录。"),
        ("user", f"账号{acc()}，注单{oid()}"),
        ("assistant", f"已核对您的注单，系统记录的下注确认赔率与您描述存在差异。{R(MUST_INCLUDE_ELEMENTS['action'])}，技术团队会调取下注时间点的赔率变更日志进行核查。{R(MUST_INCLUDE_ELEMENTS['sla'])}。如确认为系统异常，会按正确赔率重新结算。{R(MUST_INCLUDE_ELEMENTS['record'])}。"),
    ], True

def must_inc_4():
    return [
        ("user", f"我的账号被限红了，下注只能下很少"),
        ("assistant", f"{R(EMPATHY)}{R(MUST_INCLUDE_ELEMENTS['verify'])}，我帮您查看当前的限额状态和原因。"),
        ("user", f"账号{acc()}"),
        ("assistant", f"已查看您的账户，系统风控模型对您的投注做了限额调整。{R(MUST_INCLUDE_ELEMENTS['action'])}，提交了限额申诉。运营团队会调取您的操作日志核实，{R(MUST_INCLUDE_ELEMENTS['sla'])}。复核期间现有限额仍生效，核实通过后会恢复正常额度。{R(CLOSING)}"),
    ], True

def must_inc_5():
    return [
        ("user", f"我的{R(PARLAYS)}怎么还没结算？比赛都结束了"),
        ("assistant", f"{R(EMPATHY)}{R(MUST_INCLUDE_ELEMENTS['verify'])}，我帮您查看串关中各场赛果确认情况。"),
        ("user", f"账号{acc()}，注单{oid()}"),
        ("assistant", f"已查到您的串关注单，其中有一场赛事赛果与数据源同步延迟，导致整单无法结算。{R(MUST_INCLUDE_ELEMENTS['action'])}，{R(MUST_INCLUDE_ELEMENTS['sla'])}。结算完成后金额会自动到您的账户余额。{R(CLOSING)}"),
    ], True

INTENT_TEMPLATES = {
    "赛事变更": [match_change_1, match_change_2, match_change_3, match_change_4, match_change_5, match_change_6, match_change_7, match_change_8, match_change_9, match_change_10],
    "升级场景强化": [esc_1, esc_2, esc_3, esc_4, esc_5, esc_6, esc_7, esc_8, esc_9, esc_10],
    "必含要素强化": [must_inc_1, must_inc_2, must_inc_3, must_inc_4, must_inc_5],
}

def generate():
    all_samples = []
    stats = Counter()
    for intent, count in QUOTA.items():
        funcs = INTENT_TEMPLATES[intent]
        for i in range(count):
            fn = funcs[i % len(funcs)]
            try:
                msgs_tuples, esc = fn()
                messages = [{"role": r, "content": c} for r, c in msgs_tuples]
                merged = [messages[0]]
                for m in messages[1:]:
                    if merged[-1]["role"] == m["role"]:
                        merged[-1]["content"] += "\n" + m["content"]
                    else:
                        merged.append(m)
                if len(merged) >= 2 and merged[0]["role"] == "user":
                    all_samples.append({"id": rid(), "split": "train", "messages": merged})
                    stats[intent] += 1
            except Exception as e:
                print(f"  [WARN] {intent} fn={fn.__name__}: {e}", file=sys.stderr)
    random.shuffle(all_samples)
    print(f"Generated {len(all_samples)} samples")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return all_samples

if __name__ == "__main__":
    out_dir = os.path.expanduser("~/qwen35a3b_finetune/datasets/v4_fix_pack")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sft_v4_fix_6k.jsonl")
    samples = generate()
    with open(out_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nOutput: {out_file}")
    print(f"Total: {len(samples)}")
    merge_file = os.path.expanduser("~/qwen35a3b_finetune/datasets/sft_merged_v4_with_fix.jsonl")
    existing = os.path.expanduser("~/qwen35a3b_finetune/datasets/sft_merged_v3_with_boost.jsonl")
    if os.path.exists(existing):
        print(f"\nMerging with existing: {existing}")
        with open(existing, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
        with open(merge_file, "w", encoding="utf-8") as f:
            for line in existing_lines:
                f.write(line)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        total = len(existing_lines) + len(samples)
        print(f"Merged: {len(existing_lines)} + {len(samples)} = {total}")
        print(f"Output: {merge_file}")
    else:
        print(f"\n[WARN] Existing data not found: {existing}")
        print(f"Please manually merge with your v3 data.")
