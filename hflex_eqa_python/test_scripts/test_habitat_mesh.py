import math

import habitat_sim
import matplotlib.pyplot as plt
from magnum import Vector3


def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * f)))


def _make_sensor(sensor_type, width=640, height=360, f=90.0, camera_height=0.0):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = str(sensor_type)
    spec.sensor_type = sensor_type
    spec.resolution = [height, width]
    spec.position = Vector3(0.0, camera_height, 0.0)
    spec.orientation = Vector3(0.0, 0.0, 0.0)
    spec.hfov = f_to_hfov(f, width)
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return spec


sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.gpu_device_id = -1
sim_cfg.enable_physics = False
sim_cfg.allow_sliding = False
sim_cfg.scene_id = "/media/albert/ExtremeAlbert3/habitat-sim-data/versioned_data/hm3d-0.2/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb"
sim_cfg.scene_dataset_config_file = "/media/albert/ExtremeAlbert3/habitat-sim-data/versioned_data/hm3d-0.2/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json"

agent_config = habitat_sim.AgentConfiguration()
agent_config.height = 0.5
agent_config.radius = 0.1
agent_config.sensor_specifications = [
    _make_sensor(x, camera_height=0.5, width=640, height=480, f=525.0)
    for x in [
        habitat_sim.SensorType.COLOR,
        habitat_sim.SensorType.DEPTH,
        habitat_sim.SensorType.SEMANTIC,
    ]
]


config = habitat_sim.Configuration(sim_cfg, [agent_config])
sim = habitat_sim.Simulator(config)

observation = sim.get_sensor_observations()
depth = observation[str(habitat_sim.SensorType.DEPTH)]
rgb = observation[str(habitat_sim.SensorType.COLOR)][:, :, :3]
semantics = observation[str(habitat_sim.SensorType.SEMANTIC)]
