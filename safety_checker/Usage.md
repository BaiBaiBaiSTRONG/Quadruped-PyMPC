在你的主程序中调用这个安全检查器：

```python
from safety_checker import SafetyChecker
from safety_checker.core.safety_level import SafetyLevel

class YourRobotController:
    def __init__(self):
        # 定义状态观测量名称
        state_observables = (
            'base_pos', 'base_lin_vel', 'base_ori_euler_xyz',
            'base_ori_quat_wxyz', 'base_ang_vel', 'qpos_js',
            'qvel_js', 'tau_ctrl_setpoint', 'feet_pos_base',
            'feet_vel_base', 'contact_state', 'contact_forces_base'
        )

        # 定义MPC相关观测量名称
        mpc_observables = (
            'ref_base_height', 'ref_base_angles', 'ref_feet_pos',
            'nmpc_GRFs', 'nmpc_footholds', 'swing_time',
            'phase_signal', 'lift_off_positions'
        )

        # 初始化安全检查器
        self.safety_checker = SafetyChecker(
            state_observable_names=state_observables,
            mpc_observable_names=mpc_observables,
            # 可选：指定自定义配置文件路径
            config_path="path/to/your/safety_config.yaml"
        )

    def control_loop(self):
        while True:
            # 获取当前状态
            current_state = self.get_robot_state()

            # 基本用法：只获取安全等级
            safety_level = self.safety_checker.check_safety(current_state)

            if safety_level == SafetyLevel.EMERGENCY:
                self.emergency_stop()
            elif safety_level == SafetyLevel.DANGER:
                self.reduce_speed()
            elif safety_level == SafetyLevel.WARNING:
                self.cautious_mode()

            # 高级用法：获取详细信息
            safety_status = self.safety_checker.check_safety(
                current_state,
                return_details=True
            )

            # 获取具体的违反约束信息
            for constraint, violation in safety_status.violations.items():
                print(f"Constraint {constraint} violated by {violation}")

            # 获取建议动作
            print(f"Recommended action: {safety_status.recommended_action}")

            # 获取统计信息
            stats = self.safety_checker.get_statistics()

            # 正常的控制逻辑
            self.execute_control()

    def get_robot_state(self) -> dict:
        """获取机器人当前状态"""
        # 示例状态字典
        return {
            'base_pos': np.array([0.0, 0.0, 0.5]),
            'base_lin_vel': np.array([0.1, 0.0, 0.0]),
            'base_ori_euler_xyz': np.array([0.0, 0.0, 0.0]),
            'base_ori_quat_wxyz': np.array([1.0, 0.0, 0.0, 0.0]),
            'base_ang_vel': np.array([0.0, 0.0, 0.0]),
            'qpos_js': np.zeros(12),  # 12个关节角度
            'qvel_js': np.zeros(12),  # 12个关节速度
            'tau_ctrl_setpoint': np.zeros(12),
            'feet_pos_base': np.zeros((4, 3)),
            'feet_vel_base': np.zeros((4, 3)),
            'contact_state': np.array([1, 1, 1, 1]),
            'contact_forces_base': np.zeros((4, 3)),
            'ref_base_height': 0.5,
            'ref_base_angles': np.array([0.0, 0.0, 0.0]),
            'ref_feet_pos': np.zeros((4, 3)),
            'nmpc_GRFs': np.zeros((4, 10, 3)),  # 假设预测时域为10
            'nmpc_footholds': np.zeros((4, 10, 3)),
            'swing_time': np.array([0.0, 0.0, 0.0, 0.0]),
            'phase_signal': np.array([0.0, 0.0, 0.0, 0.0]),
            'lift_off_positions': np.zeros((4, 3))
        }
```

一些补充说明：

1. 配置文件的放置：

```
your_project/
    ├── config/
    │   └── safety_config.yaml    # 你的配置文件
    ├── your_controller.py        # 主控制器
    └── requirements.txt          # 包含safety_checker的依赖
```

2. 运行时变量的查看：

```python
# 检查当前激活的约束
active_constraints = safety_checker.get_active_constraints()

# 查看约束权重
scheduler_stats = safety_checker.scheduler.current_weights

# 重置安全检查器（如果需要）
safety_checker.reset()
```

3. 性能监控：

```python
# 定期检查安全统计
safety_stats = safety_checker.get_statistics()
violation_frequencies = safety_stats['violation_frequencies']
safety_levels = safety_stats['safety_levels']
```

4. 异步使用（如果需要）：

```python
import asyncio

async def safety_monitoring_loop():
    while True:
        state = get_robot_state()
        safety_status = safety_checker.check_safety(state, return_details=True)
        if safety_status.level <= SafetyLevel.DANGER:
            await handle_safety_violation(safety_status)
        await asyncio.sleep(0.01)  # 100Hz
```
