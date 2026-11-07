class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 10
    cost_threshold = 20.0
    cost_barrier = 50.0  # 与 Building 建图配置的障碍代价一致。


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()
