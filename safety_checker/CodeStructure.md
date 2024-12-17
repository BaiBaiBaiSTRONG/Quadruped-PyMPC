SafetyChecker/
├── core/
│ ├── barrier_functions.py
│ ├── safety_level.py
│ └── smooth_functions.py
│
├── constraints/
│ ├── base_constraints.py
│ ├── leg_constraints.py
│ ├── grf_constraints.py
│ ├── tracking_constraints.py
│ └── prediction_constraints.py
│
├── scheduler/
│ ├── constraint_scheduler.py
│ └── activation_optimizer.py # 新增：约束激活优化器
│
└── safety_checker.py # 主类文件
