# config_utils.py
# 该模块负责加载、合并和管理项目的配置信息。
# 主要功能包括：
# 1. 从YAML文件加载配置。
# 2. 支持全局设置和活动配置块的层级结构。
# 3. 将命令行参数与YAML配置进行合并，命令行参数具有最高优先级。
# 4. 提供一个ConfigObject类，使得配置项可以通过属性点号访问。

import yaml
import argparse
import os
import collections.abc

# 使用自定义类以支持点属性访问，同时仍可作为字典使用
class ConfigObject(dict):
    """
    配置对象类，继承自dict，提供通过属性访问配置项的能力。

    该类旨在将嵌套的字典转换为可以通过点操作符 (e.g., `config.train.batch_size`)
    进行访问的对象结构，同时保留了字典原有的方法。
    当访问一个值为字典的属性时，该字典也会被自动转换为ConfigObject实例，
    从而实现深层次的属性访问。
    """
    # Store the name of the active config block if loaded
    _active_config_name = None

    def __getattr__(self, name):
        """
        当尝试通过属性点号访问配置项时调用。

        如果 `name` 是字典中的一个键，则返回其对应的值。
        如果该值本身是一个字典但尚未转换为 `ConfigObject`，则会先进行转换。
        如果 `name` 不是一个有效的键，则抛出 `AttributeError`。

        Args:
            name (str): 尝试访问的属性名。

        Returns:
            any: 对应属性的值。如果原值为字典，则返回 `ConfigObject` 实例。

        Raises:
            AttributeError: 如果配置中不存在名为 `name` 的属性。
        """
        if name in self:
            value = self[name]
            # 动态转换嵌套字典为ConfigObject
            if isinstance(value, dict) and not isinstance(value, ConfigObject):
                value = ConfigObject(value)
                self[name] = value # 更新，以便下次直接访问
            return value
        else:
            # 对于不存在的属性，明确抛出AttributeError，行为与普通对象一致
            # Allow access to internal attributes like _active_config_name
            if name == '_active_config_name':
                 return self.__dict__.get(name)
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def __setattr__(self, name, value):
        """
        当尝试通过属性点号设置配置项时调用。

        Args:
            name (str): 要设置的属性名。
            value (any): 要设置的属性值。
        """
        # Allow setting internal attributes
        if name == '_active_config_name':
            self.__dict__[name] = value
        else:
            self[name] = value

    def __delattr__(self, name):
        """
        当尝试通过属性点号删除配置项时调用。

        Args:
            name (str): 要删除的属性名。

        Raises:
            AttributeError: 如果配置中不存在名为 `name` 的属性。
        """
        if name == '_active_config_name':
             if name in self.__dict__:
                 del self.__dict__[name]
             else:
                 raise AttributeError(f"配置中不存在内部属性: '{name}'")
        elif name in self:
            del self[name]
        else:
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def get(self, key, default=None):
        """
        获取配置项，如果键不存在则返回指定的默认值。

        如果获取到的值是字典，则会确保其作为 `ConfigObject` 返回。

        Args:
            key (str): 要获取的配置项的键。
            default (any, optional): 如果键不存在时返回的默认值。默认为 `None`。

        Returns:
            any: 对应键的值，或默认值。如果原值为字典，则返回 `ConfigObject`。
        """
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, ConfigObject):
            # Ensure nested dictionaries obtained via get() are also ConfigObjects
            value = ConfigObject(value)
            # Optionally update the main dict? No, get should be read-only
        return value

    def to_dict(self):
        """
        将ConfigObject实例及其所有嵌套的ConfigObject转换为普通的Python字典。

        这在需要将配置传递给不支持自定义对象序列化或需要标准字典格式的
        外部库或函数时非常有用。

        Returns:
            dict: 包含所有配置项的标准Python字典。
        """
        regular_dict = {}
        for key, value in self.items():
            # Skip internal attributes like _active_config_name
            # if key.startswith('_'):
            #     continue
            if isinstance(value, ConfigObject):
                regular_dict[key] = value.to_dict() # 递归转换
            elif isinstance(value, dict): # 处理普通字典（理论上应该在访问时已转为ConfigObject）
                inner_dict = {}
                for k_inner, v_inner in value.items():
                    if isinstance(v_inner, ConfigObject):
                        inner_dict[k_inner] = v_inner.to_dict()
                    else:
                        inner_dict[k_inner] = v_inner
                regular_dict[key] = inner_dict
            else:
                regular_dict[key] = value
        return regular_dict

# 辅助函数：深层合并字典 (to_merge 会覆盖 base 中的值)
def deep_update(base: dict, to_merge: dict) -> dict:
    """
    递归地更新（合并）字典。

    将 `to_merge` 字典中的键值对合并到 `base` 字典中。
    如果某个键在 `base` 和 `to_merge` 中都存在，并且其值都是字典，
    则会递归地对这两个内层字典进行合并。
    否则，`to_merge` 中的值会直接覆盖 `base` 中的值。

    Args:
        base (dict): 基础字典，将被更新。
        to_merge (dict): 包含要合并的键值对的字典。

    Returns:
        dict: 更新后的 `base` 字典。
    """
    for key, value in to_merge.items():
        if isinstance(value, collections.abc.Mapping) and key in base and isinstance(base.get(key), collections.abc.Mapping):
            # 如果键存在且对应的值都是字典（或类似映射的对象），则递归更新
            # 使用 base.get(key, {}) 确保即使 base[key] 是 None 或其他非映射类型也能处理
            base[key] = deep_update(base.get(key, {}), value)
        else:
            # 否则，直接用 to_merge 中的值覆盖 base 中的值
            base[key] = value
    return base

def load_config(default_yaml_path: str, cmd_args_namespace: argparse.Namespace) -> ConfigObject:
    """
    加载项目配置，遵循特定的优先级顺序：
    1. 命令行参数 (cmd_args_namespace 中非 None 的值)
    2. 活动配置块 (由命令行 --active_config 或 YAML active_config 指定)
    3. 全局设置 (YAML 中的 global_settings)
    4. YAML 文件中的其他顶级设置 (作为后备)

    Args:
        default_yaml_path (str): 当命令行未通过 `--config_path` 指定配置文件时，
                                 使用的默认YAML配置文件路径。
        cmd_args_namespace (argparse.Namespace): 由 `argparse` 解析后的命令行参数对象。
                                                期望包含一个可选的 `config_path` 属性，
                                                以及其他可能覆盖YAML配置的参数。

    Returns:
        ConfigObject: 一个包含最终合并后所有配置项的 `ConfigObject` 实例。
                      可以通过点属性方式访问配置内容。
    """
    # 确定最终要加载的YAML文件路径
    yaml_file_to_load = getattr(cmd_args_namespace, 'config_path', None) or default_yaml_path

    full_yaml_config = {}
    if yaml_file_to_load and os.path.exists(yaml_file_to_load):
        # print(f"从 YAML 文件加载配置: {yaml_file_to_load}") # 注释掉
        try:
            with open(yaml_file_to_load, 'r', encoding='utf-8') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml:
                    full_yaml_config = loaded_yaml
                else:
                    print(f"警告: YAML 配置文件 {yaml_file_to_load} 为空或无效。")
        except yaml.YAMLError as e:
            print(f"错误: 解析 YAML 文件 {yaml_file_to_load} 失败: {e}")
        except Exception as e:
            print(f"错误: 读取 YAML 文件 {yaml_file_to_load} 时发生未知错误: {e}")
    elif yaml_file_to_load:
        print(f"警告: 指定的 YAML 配置文件 {yaml_file_to_load} 未找到。")
    else:
        print("未提供 YAML 配置文件路径。")

    # 步骤 1: 获取全局设置
    global_settings = full_yaml_config.get('global_settings', {})
    merged_config_dict = global_settings.copy()

    # 步骤 2: 确定并获取活动配置块名称
    cmd_active_config = getattr(cmd_args_namespace, 'active_config', None)
    active_config_name_to_use = cmd_active_config # 命令行优先
    if active_config_name_to_use is None:
        active_config_name_to_use = full_yaml_config.get('active_config')

    # 步骤 3: 加载并深层合并活动配置块到全局设置中
    if active_config_name_to_use:
        active_config_block = full_yaml_config.get(active_config_name_to_use)
        if active_config_block and isinstance(active_config_block, dict):
            # print(f"加载并合并活动配置块: {active_config_name_to_use}") # 注释掉
            merged_config_dict = deep_update(merged_config_dict, active_config_block)
        elif active_config_block:
             print(f"警告: 活动配置块 '{active_config_name_to_use}' 的值不是一个有效的配置字典，已忽略。")
        elif full_yaml_config:
            print(f"警告: 在YAML中找不到名为 '{active_config_name_to_use}' 的配置块。")
    elif cmd_active_config:
         print(f"警告: 命令行指定的 active_config '{cmd_active_config}' 在YAML中找不到。")
    else:
        print("提示: 未指定 active_config。将仅使用全局设置（如果存在）和命令行参数。")
    
    # 步骤 3.5 (新增): 如果指定了 active_infer_config，将其参数合并到最终配置中 (最高优先级)
    cmd_active_infer_config = getattr(cmd_args_namespace, 'active_infer_config', None)
    active_infer_config_name_to_use = cmd_active_infer_config
    if active_infer_config_name_to_use is None:
        active_infer_config_name_to_use = full_yaml_config.get('active_infer_config')

    if active_infer_config_name_to_use:
        infer_config_block = full_yaml_config.get(active_infer_config_name_to_use)
        if infer_config_block and isinstance(infer_config_block, dict):
            # print(f"加载并合并推理活动配置块: {active_infer_config_name_to_use}") # 注释掉
            # 深层合并推理配置，覆盖之前的设置
            merged_config_dict = deep_update(merged_config_dict, infer_config_block)
        elif infer_config_block:
            print(f"警告: 推理活动配置块 '{active_infer_config_name_to_use}' 的值不是一个有效的配置字典，已忽略。")
        elif full_yaml_config:
            print(f"警告: 在YAML中找不到名为 '{active_infer_config_name_to_use}' 的推理配置块。")
    elif cmd_active_infer_config:
        print(f"警告: 命令行指定的 active_infer_config '{cmd_active_infer_config}' 在YAML中找不到。")
    
    # 步骤 4: 合并命令行参数 (最高优先级)
    cmd_args_dict = vars(cmd_args_namespace)
    final_merged_config = merged_config_dict # 从已合并YAML配置开始

    # print("--- 合并命令行参数 (覆盖YAML) ---") # 注释掉
    for key, cmd_value in cmd_args_dict.items():
        # 跳过仅用于加载过程的参数
        if key in ['config_path', 'active_config']:
            continue

        # 只有当命令行参数提供了非None值时，才用它来覆盖
        if cmd_value is not None:
            # 检查是否尝试覆盖字典
            if key in final_merged_config and isinstance(final_merged_config[key], dict) and not isinstance(cmd_value, dict):
                 print(f"  警告: 命令行参数 '{key}' (值: {cmd_value}) 正在覆盖一个字典类型的配置项。")
            
            # 直接覆盖或添加
            final_merged_config[key] = cmd_value
            # print(f"  配置项 '{key}' 已被命令行值覆盖: {cmd_value}") # 注释掉
        # 如果命令行参数是 None，我们不覆盖已有的值
        # 但如果这个键之前完全不存在，可以考虑添加（虽然 argparse 通常会处理默认值）
        elif key not in final_merged_config:
             final_merged_config[key] = cmd_value # 添加 YAML 中没有但命令行中为 None 的键
             # print(f"  配置项 '{key}' (值为None) 已从命令行添加。") # 注释掉
    # print("-----------------------------------") # 注释掉

    # 将最终合并的字典转换为ConfigObject实例
    final_config_obj = ConfigObject(final_merged_config)
    # 存储活动配置名称（如果有的话），以便在其他地方引用
    if active_config_name_to_use:
        # Use __setattr__ to set the internal attribute
        final_config_obj._active_config_name = active_config_name_to_use
        final_config_obj.active_config = active_config_name_to_use # Make it accessible at top level
    
    # NEW: Store the active_infer_config_name at the top level for infer.py to use
    if active_infer_config_name_to_use: # 检查 active_infer_config_name_to_use 是否有值
        final_config_obj.active_infer_config = active_infer_config_name_to_use
    
    # 打印最终生效的配置项，方便调试和确认
    # print("--- 最终生效的配置项 ---") # 注释掉
    def _print_config_recursively(item: dict, indent_level: int = 0):
        """内部辅助函数，递归打印配置项，保持缩进美观。"""
        indent_str = "  " * indent_level
        # 对键进行排序，确保打印顺序一致，便于比较不同运行间的配置差异
        for k, v in sorted(item.items()):
            # Skip internal attributes when printing the final config
            # if k.startswith('_'):
            #     continue
            if isinstance(v, ConfigObject) or isinstance(v, dict): # ConfigObject也是dict的子类
                print(f"{indent_str}{k}:")
                _print_config_recursively(v, indent_level + 1)
            else:
                print(f"{indent_str}{k}: {v}")
    
    # Pass the ConfigObject itself to the print function
    # _print_config_recursively(final_config_obj, indent_level=1) # 注释掉
    # print("-----------------------------------------") # 注释掉

    return final_config_obj
