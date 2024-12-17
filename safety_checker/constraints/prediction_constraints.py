import numpy as np
from typing import Dict, List, Union
from ..core.barrier_functions import BarrierFunction, BarrierConfig
from ..core.safety_level import ConstraintStatus, ConstraintPriority

class PredictionConstraint:
    """Base class for prediction-related constraints"""
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = "prediction_constraint"
        self._init_barrier_functions()

    def _init_barrier_functions(self):
        """Initialize barrier functions"""
        raise NotImplementedError

    def check(self, state_dict: Dict) -> ConstraintStatus:
        """Check constraint satisfaction"""
        raise NotImplementedError

class PredictedFootholdConstraint(PredictionConstraint):
    """Constraint on predicted foothold positions"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "predicted_foothold"
        self.num_legs = 4

    def _init_barrier_functions(self):
        self.workspace_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['workspace_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.workspace_barrier = BarrierFunction(self.workspace_barrier_config)
        
        self.step_length_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['step_length_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][1]
        )
        self.step_length_barrier = BarrierFunction(self.step_length_barrier_config)

    def check(self, state_dict: Dict) -> List[ConstraintStatus]:
        if not self.enabled:
            return [ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=False,
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        predicted_footholds = state_dict['nmpc_footholds']
        current_positions = state_dict['feet_pos_base']
        
        # 确保数据格式正确
        leg_order = ['FL', 'FR', 'RL', 'RR']
        
        statuses = []
        for i, leg in enumerate(leg_order):
            try:
                # 确保 predicted_footholds_dict[leg] 是二维数组
                foothold_predictions = np.array(predicted_footholds[leg])
                if len(foothold_predictions.shape) == 1:
                    foothold_predictions = foothold_predictions.reshape(1, -1)
                    
                # 计算距离
                distances = np.linalg.norm(foothold_predictions, axis=1)
                max_distance = np.max(distances)
                workspace_value = self.workspace_barrier.evaluate(max_distance)
                
                # 确保 current_positions 格式正确
                current_pos = np.array(current_positions[i])
                if len(current_pos.shape) == 1:
                    current_pos = current_pos.reshape(1, -1)
                    
                # 计算步长
                step_lengths = np.linalg.norm(
                    foothold_predictions - current_pos,
                    axis=1
                )
                max_step_length = np.max(step_lengths)
                step_length_value = self.step_length_barrier.evaluate(max_step_length)
                
                value = min(workspace_value, step_length_value)
                threshold = min(
                    self.workspace_barrier_config.threshold,
                    self.step_length_barrier_config.threshold
                )
                
                statuses.append(ConstraintStatus(
                    name=f"{self.name}_leg_{leg}",
                    enabled=True,
                    priority=ConstraintPriority.HIGH,
                    value=value,
                    threshold=threshold,
                    violation=max(0, threshold - value),
                    is_violated=value < 0
                ))
                
            except Exception as e:
                print(f"处理腿 {leg} 时出错: {str(e)}")
                print(f"predicted_footholds shape: {predicted_footholds[leg].shape}")
                print(f"current_positions shape: {current_positions[i].shape}")
                raise
                
        return statuses

class PredictedGRFConstraint(PredictionConstraint):
    """Constraint on predicted ground reaction forces"""
    def __init__(self, config: Dict):
        self.name = "predicted_grf"
        self.num_legs = 4
        self.friction_coef = config['thresholds']['friction_coefficient']
        super().__init__(config)

    def _init_barrier_functions(self):
        self.force_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['force_max'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.force_barrier = BarrierFunction(self.force_barrier_config)
        
        self.friction_barrier_config = BarrierConfig(
            threshold=np.arctan(self.friction_coef),
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][1]
        )
        self.friction_barrier = BarrierFunction(self.friction_barrier_config)

    def check(self, state_dict: Dict) -> List[ConstraintStatus]:
        if not self.enabled:
            return [ConstraintStatus(
                name=f"{self.name}_leg_{i}",
                enabled=False,
                priority=ConstraintPriority.HIGH,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            ) for i in range(self.num_legs)]

        predicted_grfs = state_dict['nmpc_GRFs']
        leg_order = ['FL', 'FR', 'RL', 'RR']
        
        statuses = []
        for leg in leg_order:
            try:
                # 获取当前腿的GRF数据
                leg_grfs = np.array(predicted_grfs[leg])
                if len(leg_grfs.shape) == 1:
                    leg_grfs = leg_grfs.reshape(1, -1)
                
                # 检查力的大小约束
                force_magnitudes = np.linalg.norm(leg_grfs, axis=1)
                max_force = np.max(force_magnitudes)
                force_value = self.force_barrier.evaluate(max_force)
                
                # 检查摩擦锥约束
                force_angles = np.arctan2(
                    np.linalg.norm(leg_grfs[:, :2], axis=1),
                    leg_grfs[:, 2]
                )
                max_angle = np.max(force_angles)
                friction_value = self.friction_barrier.evaluate(max_angle)
                
                value = min(force_value, friction_value)
                threshold = min(
                    self.force_barrier_config.threshold,
                    self.friction_barrier_config.threshold
                )

                statuses.append(ConstraintStatus(
                    name=f"{self.name}_leg_{leg}",
                    enabled=True,
                    priority=ConstraintPriority.HIGH,
                    value=value,
                    threshold=threshold,
                    violation=max(0, threshold - value),
                    is_violated=value < 0
                ))
                
            except Exception as e:
                print(f"处理腿 {leg} 的GRF时出错: {str(e)}")
                print(f"predicted_grfs shape for leg {leg}: {predicted_grfs[leg].shape}")
                raise

        return statuses

class PredictedStabilityConstraint(PredictionConstraint):
    """Constraint on predicted stability over prediction horizon"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "predicted_stability"

    def _init_barrier_functions(self):
        self.com_barrier_config = BarrierConfig(
            threshold=self.config['thresholds']['com_margin_min'],
            warning_threshold=self.config['thresholds']['warning'],
            danger_threshold=self.config['thresholds']['danger'],
            emergency_threshold=self.config['thresholds']['emergency'],
            weight=self.config['weights'][0]
        )
        self.com_barrier = BarrierFunction(self.com_barrier_config)

    def _compute_support_polygon(self, foot_positions: np.ndarray, contact_state: np.ndarray) -> np.ndarray:
        """Compute support polygon from foot positions and contact states"""
        contact_positions = foot_positions[contact_state.astype(bool)]
        if len(contact_positions) < 3:
            return np.zeros((0, 2))  # Return empty polygon if not enough contacts
        return contact_positions[:, :2]  # Return xy coordinates of contact points

    def _point_to_polygon_distance(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Compute minimum distance from point to polygon"""
        if len(polygon) == 0:
            return 0.0
        return np.min(np.linalg.norm(polygon - point, axis=1))

    def check(self, state_dict: Dict) -> ConstraintStatus:
        if not self.enabled:
            return ConstraintStatus(
                name=self.name,
                enabled=False,
                priority=ConstraintPriority.CRITICAL,
                value=0.0,
                threshold=float('inf'),
                violation=0.0,
                is_violated=False
            )

        predicted_footholds = state_dict['nmpc_footholds']  
        phase_signal = state_dict['phase_signal']           
        base_pos = state_dict['base_pos']                  
        
        # 获取腿的顺序
        leg_order = ['FL', 'FR', 'RL', 'RR']
        
        # 确定预测步数
        first_leg_predictions = np.array(predicted_footholds[leg_order[0]])
        if len(first_leg_predictions.shape) == 1:
            first_leg_predictions = first_leg_predictions.reshape(1, -1)
        num_predictions = len(first_leg_predictions)
        
        # Compute predicted stability margins
        stability_margins = []
        
        for t in range(num_predictions):
            # 收集所有腿在时间 t 的位置
            foot_positions = np.array([
                np.array(predicted_footholds[leg])[t] if len(np.array(predicted_footholds[leg]).shape) > 1 
                else np.array(predicted_footholds[leg])
                for leg in leg_order
            ])
            
            # 获取接触状态
            contact_states = 1 - np.array(phase_signal)  # 0 for swing, 1 for stance
            
            # 计算支撑多边形
            support_polygon = self._compute_support_polygon(
                foot_positions,
                contact_states
            )
            
            # 检查 CoM 稳定性（假设 CoM 高度恒定）
            com_xy = base_pos[:2]  # 使用当前 CoM 作为未来 CoM 的近似
            margin = self._point_to_polygon_distance(com_xy, support_polygon)
            stability_margins.append(margin)
        
        # 评估最小稳定性裕度
        min_margin = min(stability_margins) if stability_margins else 0.0
        value = self.com_barrier.evaluate(min_margin)
        threshold = self.com_barrier_config.threshold

        return ConstraintStatus(
            name=self.name,
            enabled=True,
            priority=ConstraintPriority.CRITICAL,
            value=value,
            threshold=threshold,
            violation=max(0, threshold - value),
            is_violated=value < 0
        )

class PredictionConstraintModule:
    """Module combining all prediction-related constraints"""
    def __init__(self, config: Dict):
        self.foothold_constraint = PredictedFootholdConstraint(config['predicted_foothold'])
        self.grf_constraint = PredictedGRFConstraint(config['predicted_grf'])
        self.stability_constraint = PredictedStabilityConstraint(config['predicted_stability'])
        
    def check_all(self, state_dict: Dict) -> Dict[str, Union[ConstraintStatus, List[ConstraintStatus]]]:
        """Check all prediction constraints"""
        return {
            'foothold': self.foothold_constraint.check(state_dict),
            'grf': self.grf_constraint.check(state_dict),
            'stability': self.stability_constraint.check(state_dict)
        }

    def get_most_critical(self, state_dict: Dict) -> ConstraintStatus:
        """Get the most critical constraint status"""
        all_statuses = self.check_all(state_dict)
        
        # Flatten all constraints
        all_statuses_flat = (
            all_statuses['foothold'] +
            all_statuses['grf'] +
            [all_statuses['stability']]
        )
        
        # Find most critical
        return min(all_statuses_flat, key=lambda x: x.value)