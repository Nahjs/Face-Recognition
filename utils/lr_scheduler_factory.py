# utils/lr_scheduler_factory.py
# 该工厂模块负责根据配置文件中的设置，动态创建和返回不同类型的学习率调度器实例。
# 目的是将学习率调度器的创建逻辑与主训练脚本分离，提高代码的模块化和可维护性。
#
# 支持的调度器类型包括：
# - StepDecay: 按预设的步长衰减学习率。
# - MultiStepDecay: 在预设的多个里程碑 (milestones) 处衰减学习率。
# - ExponentialDecay: 按指数方式衰减学习率。
# - ReduceOnPlateau: 当某个监控指标 (如验证集损失) 在一段时间内不再改善时，降低学习率。
# - CosineAnnealingDecay: 使用余弦退火方式调整学习率。
# - PolynomialDecay: 使用多项式衰减方式调整学习率。
# - CosineAnnealingWarmRestarts: 带热重启的余弦退火。
#
# 工厂函数 `get_lr_scheduler` 会解析配置文件中的 `lr_scheduler_type` 和相应的 `lr_scheduler_params`，
# 然后实例化对应的调度器对象。它还支持可选的 `warmup_params` 来为调度器添加预热 (warmup) 阶段。
# 如果配置的调度器类型不受支持，或者参数不正确，会抛出 ValueError。
#
# 示例配置 (在YAML文件中):\n# ```yaml
# # ...其他配置...
# learning_rate: 0.01 # 优化器的初始学习率，也会作为调度器的基础学习率
# lr_scheduler_type: "MultiStepDecay" # 或者 "StepDecay", "ReduceOnPlateau" 等
# lr_scheduler_params:
#   stepdecay: # 如果 lr_scheduler_type 是 StepDecay
#     step_size: 30
#     gamma: 0.1
#   multistepdecay: # 如果 lr_scheduler_type 是 MultiStepDecay
#     milestones: [30, 60, 90]
#     gamma: 0.1
#   exponentialdecay: # 如果 lr_scheduler_type 是 ExponentialDecay
#     gamma: 0.9
#   reduceonplateau: # 如果 lr_scheduler_type 是 ReduceOnPlateau
#     mode: 'min' # 或 'max'
#     factor: 0.1
#     patience: 10
#     # 其他 ReduceOnPlateau 参数...
#   cosineannealingdecay: # 如果 lr_scheduler_type 是 CosineAnnealingDecay
#     T_max: 100 # 最大轮数
#     eta_min: 0 # 学习率下限
#   polynomialdecay: # 如果 lr_scheduler_type 是 PolynomialDecay
#     decay_steps: 100 # 衰减步数
#     end_lr: 0 # 结束学习率
#     power: 1.0 # 多项式指数
#     cycle: False # 是否循环
#   cosineannealingwarmrestarts: # 如果 lr_scheduler_type 是 CosineAnnealingWarmRestarts
#     T_0: 10 # 第一个循环的轮数
#     T_mult: 2 # 每次重启后周期倍数
#     eta_min: 0 # 学习率下限
# warmup: # 可选的预热配置块 (如果需要Warmup)
#   use_warmup: true # 是否使用Warmup
#   warmup_steps: 500 # 预热步数
#   start_lr: 0.0001 # 预热开始的学习率
# ```

import paddle
import paddle.optimizer.lr as lr_
# 导入 LinearWarmup 调度器，它通常用于包装其他调度器
try:
    from paddle.optimizer.lr import LinearWarmup
except ImportError:
    print("警告: paddle.optimizer.lr 模块中未找到 LinearWarmup 类。Warmup 功能将无法使用。")
    LinearWarmup = None # Define a dummy class if LinearWarmup is not available


from config_utils import ConfigObject # 假设 ConfigObject 在 config_utils.py 中定义

def get_lr_scheduler(config: ConfigObject, initial_learning_rate: float, total_steps: int = None, epochs: int = None, steps_per_epoch: int = None):
    """
    根据配置对象创建并返回一个学习率调度器实例。

    Args:
        config (ConfigObject): 合并后的配置对象，包含 lr_scheduler_type 和 lr_scheduler_params。
        initial_learning_rate (float): 优化器的初始学习率。
        total_steps (int, optional): 总的训练步数。对于需要总步数的调度器（如OneCycleDecay, Warmup），此参数很重要。
                                     可以通过 epochs * steps_per_epoch 计算得到。默认为 None。
        epochs (int, optional): 总的训练 epoch 数。用于某些调度器（如CosineAnnealingDecay）。默认为 None。
        steps_per_epoch (int, optional): 每个 epoch 的总步数。用于计算 total_steps。默认为 None。

    Returns:
        paddle.optimizer.lr.LRScheduler: 创建的学习率调度器实例。

    Raises:
        ValueError: 如果配置的调度器类型不受支持，或者参数不正确/缺失。
    """
    lr_scheduler_type = config.lr_scheduler_type.lower()
    lr_scheduler_params = config.lr_scheduler_params.get(lr_scheduler_type, {})
    warmup_params = config.lr_scheduler_params.get('warmup', {})

    scheduler = None

    if lr_scheduler_type == 'stepdecay':
        step_size = lr_scheduler_params.get('step_size')
        gamma = lr_scheduler_params.get('gamma')
        if step_size is None or gamma is None:
            raise ValueError("StepDecay requires 'step_size' and 'gamma' parameters.")
        scheduler = lr_.StepDecay(
            learning_rate=initial_learning_rate,
            step_size=step_size,
            gamma=gamma
        )
        print(f"创建 StepDecay 调度器: initial_lr={initial_learning_rate}, step_size={step_size}, gamma={gamma}")

    elif lr_scheduler_type == 'multistepdecay':
        milestones = lr_scheduler_params.get('milestones')
        gamma = lr_scheduler_params.get('gamma')
        if milestones is None or gamma is None:
             raise ValueError("MultiStepDecay requires 'milestones' and 'gamma' parameters.")
        if not isinstance(milestones, list):
             raise ValueError("'milestones' for MultiStepDecay must be a list.")
        scheduler = lr_.MultiStepDecay(
            learning_rate=initial_learning_rate,
            milestones=milestones,
            gamma=gamma
        )
        print(f"创建 MultiStepDecay 调度器: initial_lr={initial_learning_rate}, milestones={milestones}, gamma={gamma}")

    elif lr_scheduler_type == 'exponentialdecay':
        gamma = lr_scheduler_params.get('gamma')
        if gamma is None:
            raise ValueError("ExponentialDecay requires 'gamma' parameter.")
        scheduler = lr_.ExponentialDecay(
            learning_rate=initial_learning_rate,
            gamma=gamma
        )
        print(f"创建 ExponentialDecay 调度器: initial_lr={initial_learning_rate}, gamma={gamma}")

    elif lr_scheduler_type == 'reduceonplateau':
        mode = lr_scheduler_params.get('mode', 'min') # 'min' or 'max'
        factor = lr_scheduler_params.get('factor', 0.1)
        patience = lr_scheduler_params.get('patience', 10)
        threshold = lr_scheduler_params.get('threshold', 1e-4)
        threshold_mode = lr_scheduler_params.get('threshold_mode', 'rel') # 'rel' or 'abs'
        cooldown = lr_scheduler_params.get('cooldown', 0)
        min_lr = lr_scheduler_params.get('min_lr', 0)
        eps = lr_scheduler_params.get('eps', 1e-8)

        scheduler = lr_.ReduceOnPlateau(
             learning_rate=initial_learning_rate,
             mode=mode,
             factor=factor,
             patience=patience,
             threshold=threshold,
             threshold_mode=threshold_mode,
             cooldown=cooldown,
             min_lr=min_lr,
             epsilon=eps # Note: Paddle's parameter is epsilon
        )
        print(f"创建 ReduceOnPlateau 调度器: initial_lr={initial_learning_rate}, mode='{mode}', factor={factor}, patience={patience}, min_lr={min_lr}")

    elif lr_scheduler_type == 'cosineannealingdecay':
        T_max = lr_scheduler_params.get('T_max') # Max number of epochs
        eta_min = lr_scheduler_params.get('eta_min', 0) # Minimum learning rate

        # CosineAnnealingDecay usually takes T_max as total steps or epochs
        # If total_steps is available, use it, otherwise use epochs
        if total_steps is not None:
             T_max_steps = total_steps
             print(f"创建 CosineAnnealingDecay 调度器: initial_lr={initial_learning_rate}, T_max (steps)={T_max_steps}, eta_min={eta_min}")
        elif epochs is not None:
             T_max_steps = epochs # Assuming T_max in config refers to epochs if total_steps not passed
             print(f"创建 CosineAnnealingDecay 调度器: initial_lr={initial_learning_rate}, T_max (epochs)={T_max_steps}, eta_min={eta_min}")
        elif T_max is not None:
             T_max_steps = T_max # Use T_max from config if epochs/total_steps not passed
             print(f"创建 CosineAnnealingDecay 调度器: initial_lr={initial_learning_rate}, T_max (from config)={T_max_steps}, eta_min={eta_min}")
        else:
             raise ValueError("CosineAnnealingDecay requires 'T_max' or 'epochs' or 'total_steps'.")


        scheduler = lr_.CosineAnnealingDecay(
            learning_rate=initial_learning_rate,
            T_max=T_max_steps,
            eta_min=eta_min
        )


    elif lr_scheduler_type == 'polynomialdecay':
         decay_steps = lr_scheduler_params.get('decay_steps')
         end_lr = lr_scheduler_params.get('end_lr')
         power = lr_scheduler_params.get('power')
         cycle = lr_scheduler_params.get('cycle', False)

         if decay_steps is None or end_lr is None or power is None:
              raise ValueError("PolynomialDecay requires 'decay_steps', 'end_lr', and 'power' parameters.")

         # PolynomialDecay takes decay_steps as total steps for decay
         # Use total_steps if available, otherwise use decay_steps from config
         if total_steps is not None:
             actual_decay_steps = total_steps
             print(f"创建 PolynomialDecay 调度器: initial_lr={initial_learning_rate}, decay_steps (total)={actual_decay_steps}, end_lr={end_lr}, power={power}, cycle={cycle}")
         else:
             actual_decay_steps = decay_steps
             print(f"创建 PolynomialDecay 调度器: initial_lr={initial_learning_rate}, decay_steps (from config)={actual_decay_steps}, end_lr={end_lr}, power={power}, cycle={cycle}")


         scheduler = lr_.PolynomialDecay(
             learning_rate=initial_learning_rate,
             decay_steps=actual_decay_steps,
             end_lr=end_lr,
             power=power,
             cycle=cycle
         )

    elif lr_scheduler_type == 'cosineannealingwarmrestarts':
        T_0 = lr_scheduler_params.get('T_0')
        T_mult = lr_scheduler_params.get('T_mult', 1)
        eta_min = lr_scheduler_params.get('eta_min', 0)

        if T_0 is None:
            raise ValueError("CosineAnnealingWarmRestarts requires 'T_0' parameter.")

        scheduler = lr_.CosineAnnealingWarmRestarts(
            learning_rate=initial_learning_rate,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        print(f"创建 CosineAnnealingWarmRestarts 调度器: initial_lr={initial_learning_rate}, T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")


    else:
        raise ValueError(f"不支持的学习率调度器类型: {lr_scheduler_type}")

    # 最后应用 Warmup 阶段 (如果启用且可用)
    if LinearWarmup and warmup_params.get('use_warmup', False):
        warmup_ratio = warmup_params.get('warmup_ratio')
        start_lr = warmup_params.get('start_lr')

        if warmup_ratio is None or start_lr is None:
            raise ValueError("Warmup configuration requires 'warmup_ratio' and 'start_lr'.")

        # 只有当总步数大于10时才启用预热，并根据比例计算预热步数
        if total_steps is not None and total_steps > 10:
            calculated_warmup_steps = int(total_steps * warmup_ratio)
            if calculated_warmup_steps == 0 and total_steps > 0: # 确保至少有1步预热如果total_steps > 0
                calculated_warmup_steps = 1
            
            print(f"为已创建的调度器 ({lr_scheduler_type.upper()}) 添加 Warmup 阶段: 总步数={total_steps}, 预热比例={warmup_ratio}, 计算预热步数={calculated_warmup_steps}, 开始学习率={start_lr}, 结束学习率={initial_learning_rate}")

            scheduler = LinearWarmup(
                learning_rate=scheduler, # Wrap the base scheduler
                warmup_steps=calculated_warmup_steps,
                start_lr=start_lr,
                end_lr=initial_learning_rate # Warmup ends at the initial learning rate of the base scheduler
            )
        else:
            print(f"警告: 未满足启用 Warmup 条件 (total_steps={total_steps} 不大于10)，跳过 Warmup。")

    return scheduler


# --- Example usage (for testing the factory module directly) ---
if __name__ == '__main__':
    print("--- 测试 get_lr_scheduler 工厂函数 ---")

    # 创建一个模拟的 ConfigObject 类，用于测试工厂函数
    class MockConfig:
        def __init__(self, data: dict):
            self._data = data

        def __getattr__(self, name):
            if name in self._data:
                value = self._data[name]
                # Convert nested dicts to MockConfig for attribute access
                if isinstance(value, dict):
                    return MockConfig(value)
                return value
            # Allow getting nested MockConfig objects as attributes
            for key, val in self._data.items():
                 if isinstance(val, dict):
                      nested_config = MockConfig(val)
                      if hasattr(nested_config, name):
                           return getattr(nested_config, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def get(self, key, default=None):
             return self._data.get(key, default)


    # 模拟一些常用的配置和训练参数
    mock_initial_lr = 0.1
    mock_epochs = 100
    mock_steps_per_epoch = 50 # 模拟每 epoch 有 50 个 batch
    mock_total_steps = mock_epochs * mock_steps_per_epoch


    # 1. 测试 StepDecay
    print("\n1. 测试 StepDecay:")
    try:
        config_step = MockConfig({
            "lr_scheduler_type": "StepDecay",
            "lr_scheduler_params": {
                 "stepdecay": {"step_size": 30, "gamma": 0.1}, # 参数键名对应 type
                 "warmup": {"use_warmup": False} # 假设 Warmup 总是存在于 lr_scheduler_params
            }
        })
        scheduler1 = get_lr_scheduler(config_step, initial_learning_rate=mock_initial_lr)
        print(f"  调度器类型: {type(scheduler1)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 2. 测试 MultiStepDecay
    print("\n2. 测试 MultiStepDecay:")
    try:
        config_multistep = MockConfig({
            "lr_scheduler_type": "MultiStepDecay",
            "lr_scheduler_params": {
                 "multistepdecay": {"milestones": [30, 60, 90], "gamma": 0.1},
                 "warmup": {"use_warmup": False}
            }
        })
        scheduler2 = get_lr_scheduler(config_multistep, initial_learning_rate=mock_initial_lr)
        print(f"  调度器类型: {type(scheduler2)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 3. 测试 ExponentialDecay
    print("\n3. 测试 ExponentialDecay:")
    try:
        config_exp = MockConfig({
            "lr_scheduler_type": "ExponentialDecay",
            "lr_scheduler_params": {
                 "exponentialdecay": {"gamma": 0.9},
                 "warmup": {"use_warmup": False}
             }
        })
        scheduler3 = get_lr_scheduler(config_exp, initial_learning_rate=mock_initial_lr)
        print(f"  调度器类型: {type(scheduler3)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 4. 测试 ReduceOnPlateau (需要 epoch 结束时的评估指标)
    print("\n4. 测试 ReduceOnPlateau:")
    try:
        config_rop = MockConfig({
            "lr_scheduler_type": "ReduceOnPlateau",
            "lr_scheduler_params": {
                 "reduceonplateau": {"mode": 'min', "factor": 0.1, "patience": 10, "min_lr": 0.0001},
                 "warmup": {"use_warmup": False}
             }
        })
        scheduler4 = get_lr_scheduler(config_rop, initial_learning_rate=mock_initial_lr)
        print(f"  调度器类型: {type(scheduler4)}")
    except ValueError as e:
        print(f"  错误: {e}")


    # 5. 测试 CosineAnnealingDecay (通常使用 epochs)
    print("\n5. 测试 CosineAnnealingDecay (使用 epochs):")
    try:
        config_cosine = MockConfig({
            "lr_scheduler_type": "CosineAnnealingDecay",
            "lr_scheduler_params": {
                 "cosineannealingdecay": {"T_max": mock_epochs, "eta_min": 0}, # T_max 可以是 epochs
                 "warmup": {"use_warmup": False}
             }
        })
        scheduler5 = get_lr_scheduler(config_cosine, initial_learning_rate=mock_initial_lr, epochs=mock_epochs)
        print(f"  调度器类型: {type(scheduler5)}")
    except ValueError as e:
        print(f"  错误: {e}")

     # 6. 测试 CosineAnnealingDecay (使用 total_steps)
    print("\n6. 测试 CosineAnnealingDecay (使用 total_steps):")
    try:
        config_cosine_steps = MockConfig({
            "lr_scheduler_type": "CosineAnnealingDecay",
            "lr_scheduler_params": {
                 "cosineannealingdecay": {"T_max": mock_total_steps, "eta_min": 0}, # T_max 可以是 total_steps
                 "warmup": {"use_warmup": False}
             }
        })
        scheduler6 = get_lr_scheduler(config_cosine_steps, initial_learning_rate=mock_initial_lr, total_steps=mock_total_steps)
        print(f"  调度器类型: {type(scheduler6)}")
    except ValueError as e:
        print(f"  错误: {e}")


    # 7. 测试 PolynomialDecay (使用 total_steps)
    print("\n7. 测试 PolynomialDecay (使用 total_steps):")
    try:
        config_polynomial_steps = MockConfig({
            "lr_scheduler_type": "PolynomialDecay",
            "lr_scheduler_params": {
                 "polynomialdecay": {"decay_steps": mock_total_steps, "end_lr": 0.0001, "power": 1.0, "cycle": False},
                 "warmup": {"use_warmup": False}
             }
         })
        scheduler7 = get_lr_scheduler(config_polynomial_steps, initial_learning_rate=mock_initial_lr, total_steps=mock_total_steps)
        print(f"  调度器类型: {type(scheduler7)}")
    except ValueError as e:
        print(f"  错误: {e}")


     # 8. 测试 CosineAnnealingWarmRestarts (使用 steps_per_epoch 转换 T_0)
    print("\n8. 测试 CosineAnnealingWarmRestarts (T_0 in epochs, convert to steps):")
    try:
        config_warm_restarts = MockConfig({
            "lr_scheduler_type": "CosineAnnealingWarmRestarts",
            "lr_scheduler_params": {
                "cosineannealingwarmrestarts": {"T_0": 10, "T_mult": 2, "eta_min": 0}, # T_0 config in epochs
                "warmup": {"use_warmup": False}
            }
        })
        scheduler8 = get_lr_scheduler(config_warm_restarts, initial_learning_rate=mock_initial_lr, steps_per_epoch=mock_steps_per_epoch)
        print(f"  调度器类型: {type(scheduler8)}")
    except ValueError as e:
        print(f"  错误: {e}")


    # 9. 测试带 Warmup 的 StepDecay
    print("\n9. 测试带 Warmup 的 StepDecay:")
    # 假设 Warmup 步数定义在 warmup 子块中
    mock_warmup_steps = 500
    mock_warmup_start_lr = 0.001
    try:
        if LinearWarmup is None:
             print("  跳过测试: LinearWarmup 类不可用。")
        else:
            config_warmup_step = MockConfig({
                "lr_scheduler_type": "StepDecay",
                "lr_scheduler_params": {
                    "stepdecay": {"step_size": 30, "gamma": 0.1},
                    "warmup": {"use_warmup": True, "warmup_steps": mock_warmup_steps, "start_lr": mock_warmup_start_lr}
                }
            })
            # 对于 Warmup，total_steps 参数很重要，确保 warmup_steps 在总步数范围内
            scheduler9 = get_lr_scheduler(config_warmup_step, initial_learning_rate=mock_initial_lr, total_steps=mock_total_steps)
            print(f"  调度器类型 (包含 Warmup): {type(scheduler9)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 10. 测试带 Warmup 的 CosineAnnealingDecay
    print("\n10. 测试带 Warmup 的 CosineAnnealingDecay:")
    try:
        if LinearWarmup is None:
             print("  跳过测试: LinearWarmup 类不可用。")
        else:
            config_warmup_cosine = MockConfig({
                "lr_scheduler_type": "CosineAnnealingDecay",
                "lr_scheduler_params": {
                    "cosineannealingdecay": {"T_max": mock_epochs, "eta_min": 0},
                    "warmup": {"use_warmup": True, "warmup_steps": mock_warmup_steps, "start_lr": mock_warmup_start_lr}
                }
            })
            scheduler10 = get_lr_scheduler(config_warmup_cosine, initial_learning_rate=mock_initial_lr, epochs=mock_epochs, total_steps=mock_total_steps)
            print(f"  调度器类型 (包含 Warmup): {type(scheduler10)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 11. 测试带 Warmup 的 CosineAnnealingWarmRestarts
    print("\n11. 测试带 Warmup 的 CosineAnnealingWarmRestarts:")
    try:
        if LinearWarmup is None:
             print("  跳过测试: LinearWarmup 类不可用。")
        else:
            config_warmup_warm_restarts = MockConfig({
                "lr_scheduler_type": "CosineAnnealingWarmRestarts",
                "lr_scheduler_params": {
                    "cosineannealingwarmrestarts": {"T_0": 10, "T_mult": 2, "eta_min": 0}, # T_0 config in epochs
                     "warmup": {"use_warmup": True, "warmup_steps": mock_warmup_steps, "start_lr": mock_warmup_start_lr}
                }
            })
            # 对于 CosineAnnealingWarmRestarts，通常 T_0 和 Warmup_steps 都需要转换为 total_steps
            scheduler11 = get_lr_scheduler(config_warmup_warm_restarts, initial_learning_rate=mock_initial_lr, steps_per_epoch=mock_steps_per_epoch, total_steps=mock_total_steps)
            print(f"  调度器类型 (包含 Warmup): {type(scheduler11)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 12. 测试不支持的调度器类型
    print("\n12. 测试不支持的调度器类型:")
    try:
        config_unsupported = MockConfig({
            "lr_scheduler_type": "UnsupportedScheduler",
            "lr_scheduler_params": {}
        })
        get_lr_scheduler(config_unsupported, initial_learning_rate=mock_initial_lr)
    except ValueError as e:
        print(f"  成功捕获到预期错误: {e}")