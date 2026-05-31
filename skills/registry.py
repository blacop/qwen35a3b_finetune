"""Skill (Tool / Function) Registry.

每个 skill 用 @register_skill 装饰即可自动加入注册表。
proxy 调用 all_tools_schema() 即可拿到 OpenAI tools 列表。
"""
import importlib
import pkgutil
import os

_REGISTRY: dict = {}


def register_skill(cls):
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Skill {cls} must have a `name` attribute")
    _REGISTRY[name] = cls
    return cls


def all_tools_schema():
    """生成 OpenAI tools schema 列表"""
    return [{
        "type": "function",
        "function": {
            "name": cls.name,
            "description": cls.description,
            "parameters": cls.parameters,
        }
    } for cls in _REGISTRY.values()]


def get_skill(name: str):
    return _REGISTRY.get(name)


def list_skills():
    return list(_REGISTRY.keys())


def autoload_tools():
    """自动 import skills/tools/ 下所有非下划线开头的模块，触发 @register_skill"""
    pkg_path = os.path.join(os.path.dirname(__file__), "tools")
    if not os.path.isdir(pkg_path):
        return
    for _, modname, _ in pkgutil.iter_modules([pkg_path]):
        if modname.startswith("_"):
            continue
        importlib.import_module(f"skills.tools.{modname}")
