import math
import random
import csv
import time
from threading import Thread, Lock
from typing import List, Tuple, Optional, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from visualization_msgs.msg import Marker
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    Quaternion,
    PoseWithCovarianceStamped,
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import GearCommand
from autoware_vehicle_msgs.msg import Engage
from autoware_adapi_v1_msgs.msg import Route
from tier4_control_msgs.msg import GateMode

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

from tf_transformations import quaternion_from_euler

random.seed(42)
np.random.seed(42) 

print("[STARTUP] All imports successful and deterministic seeds are set!")

class PathChromosome:
    def __init__(self, states: List[Tuple[float, float, float]], directions: List[int]):
        self.states = states
        self.directions = directions
        self.genes = [s[2] for s in states[1:]]
        self.waypoints = [Point(x=float(s[0]), y=float(s[1]), z=0.0) for s in states]
        self.fitness = 0.0
        self.path_length = 0.0
        self.collision_cost = 0.0
        self.safety_cost = 0.0
        self.cte_cost = 0.0
        self.curvature_cost = 0.0
        self.ref_tracking_cost = 0.0
        self.target_speed = 0.3
        self._last_publish_time = 0.0

print("[STARTUP] PathChromosome class defined OK")

class GA_PlannerNode(Node):
    def __init__(self):
        print("[INIT] GA_PlannerNode.__init__ started")
        super().__init__("ga_planner_node")
        print("[INIT] rclpy Node created")

        from rclpy.parameter import Parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, False)])
        print("[INIT] use_sim_time set to False")

        self.REAL_CAR_MODE = False
        self.last_stable_pose = None
        self._emergency_stop_active = False
        self._goal_reached = False
        self._stop_anchor = None
        self._last_best = None
        print(f"[INIT] REAL_CAR_MODE = {self.REAL_CAR_MODE}")

        self.POPULATION_SIZE = 30
        self.GENERATIONS = 15
        self.MUTATION_RATE = 0.08
        self.WAYPOINTS_PER_PATH = 30
        self.LOOK_AHEAD_DISTANCE = 7.0
        self.LATERAL_OFFSET_RIGHT = 0.0
        print(f"[INIT] GA params: POP={self.POPULATION_SIZE}, GEN={self.GENERATIONS}, "
              f"MUT={self.MUTATION_RATE}, WPP={self.WAYPOINTS_PER_PATH}")

        self.OFFSET_X = 0.0
        self.OFFSET_Y = 0.0
        self.is_calibrated = False
        self.progress_idx = 0
        self._metrics_cte_history = []
        self._metrics_steer_history = []
        self._metrics_time_history = []
        self._metrics_fitness_history = []
        self._metrics_cycle_count = 0
        self._metrics_print_interval = 50
        self._last_traj = None
        print("[INIT] Coordinate offsets initialized to zero")

        self._data_lock = Lock()
        self.ga_thread_lock = Lock()
        self.ga_thread_running = False
        self.autonomous_enabled = True
        self.current_yaw = 0.0
        print("[INIT] Locks and yaw initialized")

        self.DRIFT_THRESHOLD = 15.0
        self.HARD_STOP_CTE = 6.0   # unconditional, immediate stop -- not subject to recalibration retries
        self.DRIFT_CONFIRM_COUNT = 5
        self.MAX_REALIGN_OFFSET = 50.0
        self._drift_counter = 0
        self._recalibration_count = 0
        self.MAX_RECALIBRATIONS = 20
        print(f"[INIT] Drift protection: threshold={self.DRIFT_THRESHOLD}m, "
              f"confirm={self.DRIFT_CONFIRM_COUNT}, max_realign={self.MAX_REALIGN_OFFSET}m")

        self.COLLISION_WEIGHT = 50.0
        self.SAFETY_BUFFER_WEIGHT = 50.0
        self.ROUGHNESS_WEIGHT = 10.0
        self.CTE_WEIGHT = 15.0
        self.CURVATURE_WEIGHT = 0.3

        self.HEADING_DEV_WEIGHT = 2.0
        self.JERK_WEIGHT = 1.0
        self.DELTA_S = 1.0
        self.WHEELBASE = 2.58
        self.SMOOTHING_STEPS = 10
        self.TRAJECTORY_SPEED = 0.3
        self.MAX_CURVATURE = 0.15
        # Hard geometric cap for path GENERATION (separate from the fitness
        # penalty threshold above). Derived from the vehicle's real minimum
        # turning radius (~4.62m rear-axle, from 10.61m curb-to-curb turning
        # diameter and 2.601m wheelbase -> max curvature ~0.216), with margin.
        # Must NOT be set to MAX_CURVATURE: that is a soft penalty threshold and
        # is deliberately conservative; using it as a generation cap makes tight
        # U-turns ungeneratable, so the GA emits a near-straight path, the MPC
        # tracks it faithfully, and the vehicle drives off the road.
        self.MAX_GEN_CURVATURE = 0.08
        self.MAX_ALLOWED_CTE = 3.0
        self._v_current = 0.0
        self.V_NOMINAL      = 0.5
        self.V_MIN          = 0.3
        self.A_LAT_MAX      = 2.0
        self.V_PLAN_HORIZON = 40.0
        self.TRAJECTORY_SPEED = self.V_NOMINAL
        print(f"[INIT] TRAJECTORY_SPEED={self.TRAJECTORY_SPEED} m/s")

        self.CTE_MODE = "NORMAL"
        self.CTE_ENTER_MILD = 1.2
        self.CTE_EXIT_MILD = 0.8
        self.CTE_ENTER_RECOVERY = 2.2
        self.CTE_EXIT_RECOVERY = 1.8
        self.STEER_SLEW_RATE_MAX = 0.08
        self._prev_steering_tire_angle = 0.0
        print(f"[INIT] Steering hysteresis: enter_mild={self.CTE_ENTER_MILD}m "
              f"exit_mild={self.CTE_EXIT_MILD}m enter_recovery={self.CTE_ENTER_RECOVERY}m "
              f"exit_recovery={self.CTE_EXIT_RECOVERY}m slew_max={self.STEER_SLEW_RATE_MAX}rad")

        self.ga_rng = random.Random(42)
        self.ga_nprng = np.random.RandomState(42)
        print("[INIT] GA-dedicated RNG seeded (isolated from global random state)")

        if self.REAL_CAR_MODE:
            self.TRAJECTORY_SPEED = 0.8
            self.V_NOMINAL = 0.8
            self.GENERATIONS = 30
            self.POPULATION_SIZE = 50
            self.SMOOTHING_STEPS = 8
            print("[INIT] ⚠️  REAL CAR MODE overrides applied")
            self.get_logger().warn("🚗 REAL CAR MODE ENABLED — SAFETY LIMITS ACTIVE")

        self.map_resolution = 0.2
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0
        self.map_height = 0
        self.occupied_cells: Set[Tuple[int, int]] = set()
        self.dist_map: Optional[np.ndarray] = None
        self.map_ready = False
        self._printed_unique = False
        print("[INIT] Map data fields initialized (map NOT ready yet)")

        self.current_pose: Optional[PoseStamped] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.ref_tree: Optional[KDTree] = None
        self.ref_points_raw: np.ndarray = np.array([])
        self.ref_points: np.ndarray = np.array([])
        self.ga_cycle = 0
        self.local_ref_points = None
        self.local_tree = None
        self.local_start_idx = 0
        self.local_end_idx = 0
        self.current_progress_idx: Optional[int] = None
        self.num_real_ref_points: int = 0
        self.last_path = None
        self.alpha = 0.8
        print("[INIT] State fields initialized")
        
        self.gate_pub = self.create_publisher(
            GateMode, '/control/gate_mode_cmd', 10)
        self._gate_timer_count = 0
        self.gate_timer = self.create_timer(1.0, self._set_gate_auto)
        print("[INIT] Gate mode publisher created — will force AUTO on startup")

        print("[CSV] Attempting to load /home/mitra/lane_centerline.csv ...")
        try:
            _csv_path = "/home/mitra/lane_centerline.csv"
            print(f"[CSV] Loading: {_csv_path}")
            df = pd.read_csv(_csv_path)
            print(f"[CSV] File read OK — shape={df.shape}, columns={list(df.columns)}")

            self.ref_points_raw = df[["x", "y", "yaw"]].values.copy()
            print(f"[CSV] ref_points_raw extracted: {len(self.ref_points_raw)} points, "
                  f"x=[{self.ref_points_raw[:,0].min():.2f}, {self.ref_points_raw[:,0].max():.2f}], "
                  f"y=[{self.ref_points_raw[:,1].min():.2f}, {self.ref_points_raw[:,1].max():.2f}]")

            raw_x = self.ref_points_raw[:, 0]
            raw_y = self.ref_points_raw[:, 1]
            n_pts = len(raw_x)
            geom_yaw = self.ref_points_raw[:, 2].copy()
            for i in range(n_pts - 1):
                dx = raw_x[i+1] - raw_x[i]
                dy = raw_y[i+1] - raw_y[i]
                if math.hypot(dx, dy) > 0.01:
                    geom_yaw[i] = math.atan2(dy, dx)
            geom_yaw[-1] = geom_yaw[-2]
            for i in range(1, n_pts - 1):
                d_prev = abs(math.atan2(math.sin(geom_yaw[i] - geom_yaw[i-1]),
                                        math.cos(geom_yaw[i] - geom_yaw[i-1])))
                d_next = abs(math.atan2(math.sin(geom_yaw[i+1] - geom_yaw[i]),
                                        math.cos(geom_yaw[i+1] - geom_yaw[i])))
                if d_prev > math.radians(10) and d_next > math.radians(10):
                    geom_yaw[i] = math.atan2(
                        math.sin(geom_yaw[i-1]) + math.sin(geom_yaw[i+1]),
                        math.cos(geom_yaw[i-1]) + math.cos(geom_yaw[i+1]))
            window = 15
            smoothed_yaw = geom_yaw.copy()
            for i in range(n_pts):
                s = max(0, i - window // 2)
                e = min(n_pts, i + window // 2 + 1)
                chunk = geom_yaw[s:e]
                smoothed_yaw[i] = math.atan2(np.mean(np.sin(chunk)),
                                              np.mean(np.cos(chunk)))
            self.ref_points_raw[:, 2] = smoothed_yaw
            print(f"[CSV] Yaw recomputed from geometry and smoothed (window={window}, spike_threshold=10°)")

            self.ref_points = self.ref_points_raw.copy()
            self.ref_points[:, 0] += self.OFFSET_X
            self.ref_points[:, 1] += self.OFFSET_Y
            print(f"[CSV] ref_points after offset ({self.OFFSET_X:.3f}, {self.OFFSET_Y:.3f}): "
                  f"{len(self.ref_points)} points")

            if self.LATERAL_OFFSET_RIGHT != 0.0 and self.ref_points.shape[1] > 2:
                import numpy as _np
                yaws = self.ref_points[:, 2]
                perp_x = _np.sin(yaws)
                perp_y = -_np.cos(yaws)
                self.ref_points[:, 0] += self.LATERAL_OFFSET_RIGHT * perp_x
                self.ref_points[:, 1] += self.LATERAL_OFFSET_RIGHT * perp_y
                print(f"[CSV] Applied LATERAL_OFFSET_RIGHT={self.LATERAL_OFFSET_RIGHT}m "
                      f"to all {len(self.ref_points)} ref points")

            self.ref_tree = KDTree(self.ref_points[:, :2])
            self.num_real_ref_points = len(self.ref_points)
            print(f"[CSV] KDTree built on {self.num_real_ref_points} reference points (2D)")

            self.get_logger().info("🗺️  CSV loaded with zero offset (absolute coords)")
            self.get_logger().info(f"Loaded GA reference path with {len(self.ref_points)} points")

            self.ref_path_pub = self.create_publisher(Path, "/debug_ref_path", 10)
            ref_path = Path()
            ref_path.header.frame_id = "map"
            ref_path.header.stamp = self.get_clock().now().to_msg()
            for pt in self.ref_points:
                ps = PoseStamped()
                ps.header.frame_id = "map"
                ps.pose.position.x = float(pt[0])
                ps.pose.position.y = float(pt[1])
                ref_path.poses.append(ps)
            self.safe_publish(self.ref_path_pub, ref_path)
            print(f"[CSV] Published {len(ref_path.poses)} poses to /debug_ref_path")
            self.get_logger().info(f"🟢 Published {len(ref_path.poses)} ref points to /debug_ref_path")

        except Exception as e:
            print(f"[CSV] ❌ FAILED to load CSV: {e}")
            self.get_logger().error(f"❌ CSV failed: {e}")

        print("[INIT] Setting up ROS QoS profiles and subscriptions ...")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(OccupancyGrid,
            "/perception/occupancy_grid_map/map", self.map_callback, qos)
        print("[INIT] Subscribed to /perception/occupancy_grid_map/map")
        try:
            from sensor_msgs.msg import PointCloud2
            self.create_subscription(
                PointCloud2, "/velodyne_points",
                self._velodyne_callback, 10)
            print("[INIT] Subscribed to /velodyne_points")
        except Exception as e:
            print(f"[INIT] velodyne subscription failed: {e}")
        self.detected_objects = []
        try:
            from autoware_perception_msgs.msg import PredictedObjects as _PO
            self.create_subscription(
                _PO, "/perception/object_recognition/objects",
                self._objects_callback, 10)
            print("[INIT] Subscribed to /perception/object_recognition/objects")
        except Exception as e:
            print(f"[INIT] Object subscription failed: {e}")

        self.create_subscription(Odometry,
            "/localization/kinematic_state", self.odom_callback, pose_qos)
        print("[INIT] Subscribed to /localization/kinematic_state")

        self.create_subscription(PoseStamped,
            "/planning/mission_planning/goal", self.goal_callback, 10)
        print("[INIT] Subscribed to /planning/mission_planning/goal")

        self.path_pub = self.create_publisher(Path, "/ga_best_path", 10)
        traj_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.traj_pub = self.create_publisher(
            Trajectory, "/planning/trajectory", traj_qos)
        self.trail_pub = self.create_publisher(Path, "/vehicle_trail", 10)
        self.trail_path = Path()
        self.trail_path.header.frame_id = "map"
        self.marker_pub = self.create_publisher(Marker, "ga_status_markers", 10)
        ctrl_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.control_pub = self.create_publisher(Control, '/ga/unused_control_cmd', ctrl_qos)
        self.gear_pub = self.create_publisher(GearCommand, '/control/command/gear_cmd', 10)
        self.engage_pub = self.create_publisher(Engage, '/autoware/engage', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/planning/mission_planning/goal', 10)
        print("[INIT] All publishers created: /ga_best_path, /trajectory, /vehicle_trail, "
              "/ga_status_markers, /control_cmd, /gear_cmd, /goal")

        self.control_timer = self.create_timer(0.02, self.publish_control)
        self.latest_cte = 0.0
        self._ctrl_print_counter = 0
        print("[INIT] Control timer set to 50Hz (0.02s)")

        self._republish_timer = self.create_timer(0.02, self._republish_last_trajectory)
        print("[INIT] Fast republish timer set to 50Hz (was 20Hz — prevents Autoware planner conflict)")

        self.engage_timer = self.create_timer(0.5, self._engage_heartbeat)
        print("[INIT] Engage heartbeat timer set to 2 Hz (faster MRM recovery)")

        self.get_logger().info("✅ GA Planner initialized with Safety Buffer")
        print("[INIT] ✅ GA_PlannerNode.__init__ complete\n")
        
        route_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Route,
            "/planning/mission_planning/route",
            self.route_callback,
            route_qos,
        )
        print("[INIT] Subscribed to /planning/mission_planning/route (TRANSIENT_LOCAL QoS)")
        from autoware_adapi_v1_msgs.msg import RouteState as _RS
        from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
        _rs_qos = QoSProfile(depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        self._route_state_pub = self.create_publisher(_RS, "/api/routing/state", _rs_qos)
        self._route_state_timer = self.create_timer(10.0, self._publish_route_state)

    def _set_gate_auto(self):
        """Publish AUTO gate mode every 1 s indefinitely.
        FIX B: Autoware may still be booting when the original 10-shot timer
        fired and stopped; the gate was never switched and stayed in EXTERNAL
        mode, silently blocking all control_cmd messages."""
        self._gate_timer_count += 1
        msg = GateMode()
        msg.data = GateMode.AUTO
        self.gate_pub.publish(msg)
        if self._gate_timer_count % 10 == 1:
            print(f"[GATE] AUTO mode heartbeat #{self._gate_timer_count}")

    def safe_publish(self, publisher, msg):
        try:
            if rclpy.ok():
                publisher.publish(msg)
        except Exception as e:
            print(f"[PUBLISH] Skipped — node shutting down: {e}")

    def map_callback(self, msg: OccupancyGrid):
        print(f"[MAP_CB] Received OccupancyGrid: "
              f"size=({msg.info.width}x{msg.info.height}), res={msg.info.resolution:.2f}, "
              f"origin=({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})")

        grid = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width)
        )
        print(f"[MAP_CB] Grid reshaped — min={grid.min()}, max={grid.max()}")

        if msg.info.width > 0 and msg.info.height > 0:
            self.map_resolution = msg.info.resolution
            self.map_origin_x = msg.info.origin.position.x
            self.map_origin_y = msg.info.origin.position.y
            self.map_width = msg.info.width
            self.map_height = msg.info.height

            grid_max = int(grid.max())
            grid_min_non_neg = int(grid[grid >= 0].min()) if np.any(grid >= 0) else 0

            if not self._printed_unique:
                print("UNIQUE VALUES:", np.unique(grid))
                self._printed_unique = True

            if grid_max <= 0:
                binary_grid = np.zeros_like(grid, dtype=np.int8)
                occ_threshold = -1
                print(f"[MAP_CB] Map is all-free/unknown (max={grid_max}) — no obstacles")
            elif grid_max == 1:
                binary_grid = np.where(grid == 1, 1, 0).astype(np.int8)
                occ_threshold = 1
                print(f"[MAP_CB] Binary map detected (max=1)")
            elif grid_max == 100:
                binary_grid = np.where(grid >= 50, 1, 0).astype(np.int8)
                occ_threshold = 50
                print(f"[MAP_CB] Standard Autoware map (max=100), threshold>=50")
            else:
                pos_vals = grid[grid > 0]
                if len(pos_vals) > 0:
                    occ_threshold = int(np.percentile(pos_vals, 70))
                    occ_threshold = max(occ_threshold, grid_max // 2)
                else:
                    occ_threshold = max(1, grid_max // 2)
                binary_grid = np.where(grid >= occ_threshold, 1, 0).astype(np.int8)
                print(f"[MAP_CB] Non-standard map (max={grid_max}) — threshold>={occ_threshold}")

            num_obstacles = int(np.sum(binary_grid))
            total_cells   = self.map_width * self.map_height
            obstacle_pct  = 100.0 * num_obstacles / max(total_cells, 1)
            print(f"[MAP_CB] binary_grid: {num_obstacles} occupied cells "
                  f"out of {total_cells} total ({obstacle_pct:.1f}%)")

            if obstacle_pct > 80.0:
                print(f"[MAP_CB] ⚠️  {obstacle_pct:.0f}% of cells occupied — "
                      f"grid looks like noise/all-unknown. Treating as OBSTACLE-FREE.")
                binary_grid = np.zeros_like(grid, dtype=np.int8)
                num_obstacles = 0

            if num_obstacles == 0:
                print("[MAP_CB] No obstacles detected — collision cost inactive.")

            self.dist_map = distance_transform_edt(1 - binary_grid)
            print(f"[MAP_CB] distance_transform_edt done — "
                  f"dist_map range=[{self.dist_map.min():.2f}, {self.dist_map.max():.2f}]")

            self.occupied_cells.clear()
            ys, xs = np.where(binary_grid == 1)
            for y, x in zip(ys, xs):
                self.occupied_cells.add((int(x), int(y)))
            print(f"[MAP_CB] occupied_cells updated: {len(self.occupied_cells)} cells")

            self.map_ready = True
            self.get_logger().info(
                f"🚀 MAP UPDATED: res={self.map_resolution:.2f}, obs={num_obstacles}"
            )
        else:
            print(f"[MAP_CB] ⚠️  Skipped — zero-size grid ({msg.info.width}x{msg.info.height})")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        print(f"[POSE_CB] PoseWithCovariance: "
              f"x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}")
        ps = PoseStamped(header=msg.header, pose=msg.pose.pose)
        self.handle_pose_stamped(ps)

    def _velodyne_callback(self, msg):
        """Receive raw LiDAR point cloud from Velodyne sensor on real vehicle."""
        pass

    def _objects_callback(self, msg):
        objs = []
        for obj in msg.objects:
            try:
                x = float(obj.kinematics.initial_pose_with_covariance.pose.position.x)
                y = float(obj.kinematics.initial_pose_with_covariance.pose.position.y)
                radius = max(float(obj.shape.dimensions.x),
                            float(obj.shape.dimensions.y)) / 2.0 + 0.3
                objs.append((x, y, radius))
            except Exception:
                pass
        self.detected_objects = objs
        if objs:
            print(f"[OBJ_CB] {len(objs)} object(s): {[(round(o[0],1),round(o[1],1),round(o[2],1)) for o in objs]}")

    def odom_callback(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.current_pose = ps

        q = ps.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        print(f"[ODOM_CB] pos=({ps.pose.position.x:.3f}, {ps.pose.position.y:.3f}), "
              f"yaw={math.degrees(self.current_yaw):.1f}°, calibrated={self.is_calibrated}")

        if not self.is_calibrated:
            print(f"[ODOM_CB] Not calibrated — calling align_reference_to_vehicle() "
                  f"(vehicle at {ps.pose.position.x:.3f}, {ps.pose.position.y:.3f})")
            self.align_reference_to_vehicle()
            if self.is_calibrated:
                print(f"[ODOM_CB] ✅ Calibration complete — "
                      f"OFFSET=({self.OFFSET_X:.3f}, {self.OFFSET_Y:.3f}) m")
                if self.goal_pose is None and self.ref_points is not None:
                    goal_idx = min(1400, len(self.ref_points) - 1)
                    last = self.ref_points[goal_idx]
                    goal_msg = PoseStamped()
                    goal_msg.header.frame_id = "map"
                    goal_msg.header.stamp = self.get_clock().now().to_msg()
                    goal_msg.pose.position.x = float(last[0])
                    goal_msg.pose.position.y = float(last[1])
                    goal_msg.pose.position.z = 0.0
                    last_yaw = float(last[2]) if self.ref_points.shape[1] > 2 else 0.0
                    goal_msg.pose.orientation.w = math.cos(last_yaw / 2.0)
                    goal_msg.pose.orientation.x = 0.0
                    goal_msg.pose.orientation.y = 0.0
                    goal_msg.pose.orientation.z = math.sin(last_yaw / 2.0)
                    self.goal_pose = goal_msg
                    self.autonomous_enabled = True
                    snap_now = int(getattr(self, '_last_snap_idx', 0))
                    print(f"[ODOM_CB] 🎯 Auto-goal set internally to ref[{goal_idx}]: "
                          f"({last[0]:.2f}, {last[1]:.2f})")
                    print(f"[ODOM_CB] 📍 Starting snap: ref[{snap_now}] — "
                          f"{'✅ GOOD (near start)' if snap_now < 20 else '⚠️  FAR FROM START — redo 2D Pose Estimate!'}")
                    self.engage_vehicle()

        self.handle_pose_stamped(ps)

    def handle_pose_stamped(self, msg: PoseStamped):
        px = msg.pose.position.x
        py = msg.pose.position.y

        if self.last_stable_pose is not None:
            jump = math.hypot(
                px - self.last_stable_pose.pose.position.x,
                py - self.last_stable_pose.pose.position.y,
            )
            if jump > 1000.0:
                print(f"[POSE_GUARD] ❌ Extreme jump ({jump:.2f}m) — skipping pose update")
                self.get_logger().warn(f"⚠️ Extreme jump ({jump:.2f}m) — skipping.")
                return
            if jump > 1.0:
                print(f"[POSE_GUARD] Large pose step: {jump:.3f}m")

        if not self.is_calibrated:
            print("[POSE_GUARD] Not calibrated yet — skipping drift check "
                  "(frame offset not applied yet)")
        elif self.ref_tree is not None:
            _pg_prev  = getattr(self, '_last_snap_idx', 0)
            _pg_start = max(0, _pg_prev - 2)
            _pg_end   = min(len(self.ref_points), _pg_prev + 80)
            _pg_pts   = self.ref_points[_pg_start:_pg_end, :2]
            _pg_dists = (_pg_pts[:,0]-px)**2 + (_pg_pts[:,1]-py)**2
            _pg_w_idx = int(np.argmin(_pg_dists))
            idx       = _pg_start + _pg_w_idx

            car_yaw_pg = getattr(self, 'current_yaw', 0.0)
            if self.ref_points.shape[1] > 2:
                cand_yaw = float(self.ref_points[idx, 2])
                yaw_err  = abs(math.atan2(math.sin(car_yaw_pg - cand_yaw),
                                          math.cos(car_yaw_pg - cand_yaw)))
                if yaw_err > math.radians(60):
                    all_yaws  = self.ref_points[:, 2]
                    all_diffs = np.abs(np.arctan2(
                        np.sin(car_yaw_pg - all_yaws),
                        np.cos(car_yaw_pg - all_yaws)))
                    all_dists = np.hypot(self.ref_points[:,0]-px,
                                         self.ref_points[:,1]-py)
                    _pg_window = 80
                    _lo = max(0, _pg_prev - 10)
                    _hi = min(len(self.ref_points) - 1, _pg_prev + _pg_window)
                    window_mask = np.zeros(len(self.ref_points), dtype=bool)
                    window_mask[_lo:_hi+1] = True
                    # 120 deg, not 60: partway through a 180 deg U-turn the
                    # vehicle heading legitimately differs from every nearby
                    # reference heading by more than 60 deg. With a 60 deg gate
                    # the mask goes empty, idx is never reassigned, and the snap
                    # freezes on a stale point for the rest of the turn.
                    mask = (all_diffs < math.radians(120)) & window_mask
                    if np.any(mask):
                        idx = int(np.argmin(np.where(mask, all_dists, np.inf)))
                    elif np.any(window_mask):
                        # Never freeze: fall back to nearest point in the
                        # forward window regardless of heading agreement.
                        idx = int(np.argmin(np.where(window_mask, all_dists, np.inf)))
                        print(f"[POSE_GUARD] ⚠️  No heading-consistent ref point in window "
                              f"— falling back to nearest forward point idx={idx}")

            if idx < _pg_prev - 2:
                idx = _pg_prev - 2
            self._last_snap_idx = int(idx)
            ref_pt = self.ref_points[idx]
            dist   = math.hypot(px - ref_pt[0], py - ref_pt[1])
            _ref_yaw_pg = float(ref_pt[2]) if self.ref_points.shape[1] > 2 else 0.0
            signed_offset = ((px - ref_pt[0]) * (-math.sin(_ref_yaw_pg))
                              + (py - ref_pt[1]) * math.cos(_ref_yaw_pg))
            _veh_yaw_pg = getattr(self, 'current_yaw', 0.0)
            _head_diff_pg = math.degrees(abs(math.atan2(
                math.sin(_veh_yaw_pg - _ref_yaw_pg),
                math.cos(_veh_yaw_pg - _ref_yaw_pg))))
            if _head_diff_pg > 90.0:
                print(f"[POSE_GUARD] ⚠️  SNAP HEADING MISMATCH: vehicle yaw="
                      f"{math.degrees(_veh_yaw_pg):.1f}° vs ref[{idx}] yaw="
                      f"{math.degrees(_ref_yaw_pg):.1f}° (diff={_head_diff_pg:.1f}°) "
                      f"— snapped to a reference point facing the opposite way; "
                      f"LEFT/RIGHT sense and CTE are unreliable here")

            print(f"[POSE_GUARD] CTE={dist:.3f}m signed={signed_offset:+.3f}m "
                  f"({'LEFT' if signed_offset > 0 else 'RIGHT'} of centerline) "
                  f"to ref[{idx}]=({ref_pt[0]:.2f},{ref_pt[1]:.2f}) "
                  f"| drift_counter={self._drift_counter} | emergency={self._emergency_stop_active}")

            # FAST, UNCONDITIONAL hard stop -- fires immediately on a single
            # reading, independent of drift_counter/recalibration attempts.
            # The softer DRIFT_THRESHOLD/recalibration path below is only
            # meant for small, likely-transient localization noise; it must
            # never be allowed to keep the vehicle actively driving/steering
            # while genuinely far off the reference path. Recalibration
            # attempts previously let the vehicle keep driving through many
            # retry cycles before finally stopping, by which point it had
            # already traveled tens of meters off-path.
            if dist > self.HARD_STOP_CTE:
                if not self._emergency_stop_active:
                    print(f"[DRIFT] 🛑 HARD STOP: dist={dist:.2f}m > {self.HARD_STOP_CTE}m "
                          f"— stopping immediately, no recalibration attempt")
                    self.get_logger().error(
                        f"🛑 HARD STOP DRIFT ({dist:.2f}m) — Emergency stop.")
                self._emergency_stop_active = True

            if dist > self.DRIFT_THRESHOLD:
                self._drift_counter += 1
                print(f"[DRIFT] dist={dist:.2f}m > threshold={self.DRIFT_THRESHOLD}m "
                      f"— counter={self._drift_counter}/{self.DRIFT_CONFIRM_COUNT}")

                if self._drift_counter >= self.DRIFT_CONFIRM_COUNT:
                    if self._recalibration_count < self.MAX_RECALIBRATIONS:
                        print(f"[DRIFT] 🚨 Confirmed — triggering re-calibration "
                              f"#{self._recalibration_count+1}/{self.MAX_RECALIBRATIONS}")
                        self.get_logger().error(
                            f"🚨 Confirmed drift ({dist:.2f}m) — Re-calibrating.")
                        self.is_calibrated = False
                        self._recalibration_count += 1
                    else:
                        print(f"[DRIFT] 🚨 Max re-calibrations reached — stopping vehicle "
                              f"(no remaining mechanism to resync to a legitimate reference point)")
                        self.get_logger().error(
                            f"🛑 Drift ({dist:.2f}m) — max re-calibrations reached, forcing stop.")
                        self._emergency_stop_active = True
                    self._drift_counter = 0

                if dist > 25.0:
                    if not self._emergency_stop_active:
                        print(f"[DRIFT] 🛑 CRITICAL: dist={dist:.2f}m > 25m — emergency stop")
                        self.get_logger().error(
                            f"🛑 CRITICAL DRIFT ({dist:.2f}m) — Emergency stop.")
                        self._emergency_stop_active = True
            else:
                self._drift_counter = 0
                _clear_thresh = max(0.5, self.HARD_STOP_CTE - 1.0)
                if dist < _clear_thresh and self._emergency_stop_active:
                    print(f"[DRIFT] ✅ CTE={dist:.2f}m < {_clear_thresh:.1f}m "
                          f"— clearing emergency stop (was stale)")
                    self._emergency_stop_active = False
                    self._stop_anchor = None
        else:
            print("[POSE_GUARD] ref_tree is None — cannot compute CTE")

        self.current_pose = msg
        self.last_stable_pose = msg

        ps_trail = PoseStamped()
        ps_trail.header = msg.header
        ps_trail.pose = msg.pose
        self.trail_path.poses.append(ps_trail)
        if len(self.trail_path.poses) > 500:
            self.trail_path.poses.pop(0)
        self.trail_path.header.stamp = self.get_clock().now().to_msg()
        self.trail_path.header.frame_id = msg.header.frame_id
        self.safe_publish(self.trail_pub, self.trail_path)
        print(f"[TRAIL] Trail: {len(self.trail_path.poses)} poses → /vehicle_trail")

        self.try_start_ga()

    def goal_callback(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        print(f"\n[GOAL_CB] ✅ Goal received: ({gx:.3f}, {gy:.3f})")
        self.goal_pose = msg
        self.trail_path.poses = []
        self.autonomous_enabled = True
        self._goal_reached = False
        self._stop_anchor = None
        self.engage_vehicle()
        self.get_logger().info("Target set: Clearing old vehicle trail.")
        print("[GOAL_CB] Old vehicle trail cleared")

        with self.ga_thread_lock:
            thread_was_running = self.ga_thread_running

        print(f"[GOAL_CB] GA thread was running: {thread_was_running}")
        if not thread_was_running:
            self.current_progress_idx = None
            self._drift_counter = 0
            print("[GOAL_CB] Progress idx and drift counter reset (fresh start)")

            if self.ref_points is not None and self.current_pose is not None:
                px = float(self.current_pose.pose.position.x)
                py = float(self.current_pose.pose.position.y)
                current_snap = int(getattr(self, '_last_snap_idx', 0))
                if current_snap > 20:
                    min_dist = float('inf')
                    best_idx = 0
                    for i in range(min(50, len(self.ref_points))):
                        d = math.hypot(self.ref_points[i, 0] - px,
                                       self.ref_points[i, 1] - py)
                        if d < min_dist:
                            min_dist = d
                            best_idx = i
                    self._last_snap_idx = best_idx
                    print(f"[GOAL_CB] ⚠️  Snap was ref[{current_snap}] — "
                          f"forced to ref[{best_idx}] (dist={min_dist:.2f}m) "
                          f"for fresh route start")
                else:
                    print(f"[GOAL_CB] ✅ Snap at ref[{current_snap}] — "
                          f"already near route start")

            self.get_logger().info("✅ Target set: KEEPING current calibration. NO RESET.")
            
    def engage_vehicle(self):
        if self.goal_pose is None or not self.autonomous_enabled:
            return

        msg = Engage()
        msg.engage = True
        self.safe_publish(self.engage_pub, msg)
        print("[ENGAGE] Vehicle engaged")
        
    def _engage_heartbeat(self):
        """Periodically re-send Engage=True, GearCommand=Drive, and GateMode=AUTO.
        FIX: When MRM fires emergency_stop, Autoware latches the behavior even
        after the condition resolves. Sending Engage=True repeatedly causes
        Autoware to re-evaluate whether the MRM trigger is still present.
        Also re-send gate AUTO every heartbeat — some Autoware builds reset the
        gate to EXTERNAL on any internal state transition."""
        if self.goal_pose is None or not self.autonomous_enabled:
            return

        eng = Engage()
        eng.engage = True
        self.safe_publish(self.engage_pub, eng)

        g = GearCommand()
        g.stamp = self.get_clock().now().to_msg()
        g.command = GearCommand.DRIVE
        self.safe_publish(self.gear_pub, g)

        gate_msg = GateMode()
        gate_msg.data = GateMode.AUTO
        self.safe_publish(self.gate_pub, gate_msg)

        if self._emergency_stop_active and not self._really_in_emergency():
            print("[ENGAGE_HB] ⚠️  Clearing stale emergency flag via heartbeat")
            self._emergency_stop_active = False

        print("[ENGAGE_HB] Engage=True + Gear=Drive + Gate=AUTO sent")

    def _really_in_emergency(self) -> bool:
        """Return True only when the vehicle is genuinely far from the reference path."""
        if self.current_pose is None or self.ref_tree is None:
            return False
        try:
            ex = self.current_pose.pose.position.x
            ey = self.current_pose.pose.position.y
            dist, _ = self.ref_tree.query([ex, ey])
            return float(dist) > self.HARD_STOP_CTE
        except Exception:
            return False

    def route_callback(self, msg):
        print(f"ROUTE RECEIVED ✅")
        print(f"  goal_pose: {msg.goal_pose}")
        print(f"  available fields: {[f for f in dir(msg) if not f.startswith('_')]}")
        print(f"  goal_pose type: {type(msg.goal_pose)}")
         
        try:
           goal_pose_raw = msg.goal_pose
           pose = PoseStamped()
           pose.header = msg.header
           pose.header.frame_id = "map"
           pose.pose = goal_pose_raw
           print(f"  goal extracted: ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f})")
           self.goal_callback(pose)
        except Exception as e:
            print(f"  ❌ route_callback failed to extract goal: {e}")

    def try_start_ga(self):
        print(f"[TRY_GA] Checking: pose={'OK' if self.current_pose else 'NONE'}, "
              f"goal={'OK' if self.goal_pose else 'NONE'}, "
              f"thread_running={self.ga_thread_running}")

        if self.current_pose is None:
            if hasattr(self, 'reference_path') and self.reference_path:
                print("[TRY_GA] ⚠️  No localization — falling back to first CSV point")
                self.get_logger().warn("⚠️ Localization found! Using first CSV point as start.")
                self.current_pose = Odometry()
                self.current_pose.pose.pose.position = self.reference_path[0]
                self.current_pose.pose.pose.orientation.w = 1.0
            else:
                print("[TRY_GA] ❌ No pose and no reference_path — cannot start GA")
                self.get_logger().error("❌ CSV file not loaded yet!")
                return

        if self.goal_pose is None:
            print("[TRY_GA] No goal yet — waiting")
            return

        with self.ga_thread_lock:
            if self.ga_thread_running:
                print("[TRY_GA] GA thread already running — skipping launch")
                return
            self.ga_thread_running = True
            print("[TRY_GA] 🚀 Launching GA thread")
            self.get_logger().info("🚀 Starting GA Thread (Forced Start)")
            t = Thread(target=self._ga_thread_wrapper, daemon=True)
            t.start()

    def _ga_thread_wrapper(self):
        print("[GA_THREAD] Thread started")
        try:
            if hasattr(self, 'gear_pub'):
                from autoware_vehicle_msgs.msg import GearCommand
                g = GearCommand()
                g.command = 2
                self.safe_publish(self.gear_pub, g)
                print("[GA_THREAD] GearCommand=Drive published")

            self.run_ga()

        except Exception as e:
            print(f"[GA_THREAD] 💥 CRASHED: {e}")
            self.get_logger().error(f"💥 GA thread crashed: {e}")
            import traceback
            tb = traceback.format_exc()
            print(f"[GA_THREAD] Traceback:\n{tb}")
            self.get_logger().error(tb)
        finally:
            with self.ga_thread_lock:
                self.ga_thread_running = False
            print("[GA_THREAD] Thread exited — ga_thread_running = False")
            self.get_logger().warn("⚠️ GA thread exited — will restart on next pose/goal update.")

    def _publish_route_state(self):
        from autoware_adapi_v1_msgs.msg import RouteState as _RS
        msg = _RS()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = 2
        self._route_state_pub.publish(msg)

    def align_reference_to_vehicle(self):
        """Align local-frame CSV path to the map frame using vehicle position + heading."""
        print("[ALIGN] align_reference_to_vehicle() called")

        if self.current_pose is None:
            print("[ALIGN] ❌ current_pose is None — aborting")
            return
        if self.ref_points_raw is None or len(self.ref_points_raw) == 0:
            print("[ALIGN] ❌ ref_points_raw is empty — aborting")
            return
        if self.is_calibrated:
            print("[ALIGN] Already calibrated — skipping")
            return

        ALIGN_SNAP_TOLERANCE = 3.0
        MAX_ALLOWED_OFFSET   = 10.0
        N_CANDIDATES = 20

        curr    = self.current_pose.pose.position
        veh_yaw = self.current_yaw
        print(f"[ALIGN] Vehicle: pos=({curr.x:.4f}, {curr.y:.4f}), yaw={math.degrees(veh_yaw):.2f}°")

        pre_check_tree = KDTree(self.ref_points[:, :2])
        pre_snap_dist, pre_snap_idx = pre_check_tree.query([curr.x, curr.y])
        pre_snap_dist = float(pre_snap_dist)
        print(f"[ALIGN] Pre-alignment snap distance: {pre_snap_dist:.3f} m")
        if pre_snap_dist < ALIGN_SNAP_TOLERANCE:
            print(f"[ALIGN] ✅ Vehicle already within {ALIGN_SNAP_TOLERANCE}m of CSV — "
                  f"no offset needed. Marking calibrated.")
            self.is_calibrated          = True
            self._drift_counter         = 0
            self._emergency_stop_active = False
            self._last_snap_idx         = int(pre_snap_idx)
            self.get_logger().info(
                f"🎯 CSV already aligned: snap={pre_snap_dist:.3f} m — no offset applied.")
            return

        snap_tree = KDTree(self.ref_points_raw[:, :2])
        n_query   = min(N_CANDIDATES, len(self.ref_points_raw))
        dists, idxs = snap_tree.query([curr.x, curr.y], k=n_query)
        print(f"[ALIGN] Top-{n_query} nearest raw CSV points (indices): {list(idxs)}")

        best_idx      = None
        best_yaw_diff = float('inf')
        for i, idx in enumerate(idxs):
            csv_yaw = float(self.ref_points_raw[idx, 2])
            diff = abs(math.atan2(math.sin(veh_yaw - csv_yaw),
                                  math.cos(veh_yaw - csv_yaw)))
            print(f"[ALIGN]   candidate idx={idx:4d}  csv_yaw={math.degrees(csv_yaw):.1f}°  "
                  f"diff={math.degrees(diff):.1f}°  dist={dists[i]:.2f}m")
            if diff < math.pi / 2 and diff < best_yaw_diff:
                best_yaw_diff = diff
                best_idx      = int(idx)

        if best_idx is None:
            best_idx = int(idxs[0])
            print(f"[ALIGN] ⚠️  No yaw-consistent candidate found — "
                  f"falling back to nearest (idx={best_idx}). "
                  f"Check that the CSV yaw column matches Autoware's yaw convention.")

        snap_x   = float(self.ref_points_raw[best_idx, 0])
        snap_y   = float(self.ref_points_raw[best_idx, 1])
        snap_yaw = float(self.ref_points_raw[best_idx, 2])
        print(f"[ALIGN] Chosen snap point [{best_idx}]: "
              f"({snap_x:.4f}, {snap_y:.4f}), yaw={math.degrees(snap_yaw):.2f}°")

        offset_x   = curr.x - snap_x
        offset_y   = curr.y - snap_y
        offset_mag = math.hypot(offset_x, offset_y)
        print(f"[ALIGN] Frame offset: dx={offset_x:.4f} m, dy={offset_y:.4f} m, "
              f"magnitude={offset_mag:.4f} m")

        if offset_mag > MAX_ALLOWED_OFFSET:
            self.get_logger().error(
                f"🚨 Computed offset {offset_mag:.2f} m > {MAX_ALLOWED_OFFSET} m limit. "
                f"This means the snap point is wrong (stale vehicle pose or bad CSV coverage). "
                f"Applying ZERO offset — CSV will be used as-is in its original frame.")
            offset_x = 0.0
            offset_y = 0.0
            offset_mag = 0.0
        elif offset_mag > 5.0:
            self.get_logger().warn(f"⚠️  Moderate frame offset ({offset_mag:.2f} m) — applying.")
        else:
            self.get_logger().info(f"✅ Small frame offset ({offset_mag:.2f} m) — applying.")

        self.OFFSET_X = offset_x
        self.OFFSET_Y = offset_y
        new_ref_points = self.ref_points_raw.copy()
        new_ref_points[:, 0] += self.OFFSET_X
        new_ref_points[:, 1] += self.OFFSET_Y
        print(f"[ALIGN] Shifted CSV range: "
              f"x=[{new_ref_points[:,0].min():.2f}, {new_ref_points[:,0].max():.2f}], "
              f"y=[{new_ref_points[:,1].min():.2f}, {new_ref_points[:,1].max():.2f}]")

        with self._data_lock:
            self.ref_points          = new_ref_points
            self.ref_tree            = KDTree(self.ref_points[:, :2])
            self.num_real_ref_points = len(new_ref_points)
        print(f"[ALIGN] ref_points + KDTree updated ({len(self.ref_points)} pts)")

        dist_snap, snapped_idx = self.ref_tree.query([curr.x, curr.y])
        dist_snap  = float(dist_snap)
        snapped_idx = int(snapped_idx)
        print(f"[ALIGN] Post-alignment snap: dist={dist_snap:.4f} m, idx={snapped_idx}")

        if dist_snap > ALIGN_SNAP_TOLERANCE:
            self.get_logger().warn(
                f"⚠️  Post-alignment snap = {dist_snap:.2f} m > {ALIGN_SNAP_TOLERANCE} m. "
                f"The CSV may not cover the vehicle's current map position. "
                f"Snap point: {self.ref_points[snapped_idx, :2]}. "
                f"Check lane_centerline.csv covers this area.")
        else:
            self.get_logger().info(
                f"🎯 Alignment OK: snap={dist_snap:.3f} m at idx={snapped_idx} ✅")

        self.is_calibrated          = True
        self._drift_counter         = 0
        self._emergency_stop_active = False
        self._last_snap_idx         = int(snapped_idx)
        print(f"[ALIGN] ✅ is_calibrated=True, _last_snap_idx={snapped_idx}, drift reset")
        self.get_logger().info(
            f"🗺️  AUTO-ALIGNED: offset=({self.OFFSET_X:.3f}, {self.OFFSET_Y:.3f}) m  "
            f"magnitude={offset_mag:.3f} m  snap={dist_snap:.3f} m  "
            f"yaw_diff={math.degrees(best_yaw_diff):.1f}°"
        )

    def calculate_environmental_costs(self, path: PathChromosome):
        COLLISION_THRESHOLD = 0.8
        SAFETY_BUFFER = 2.0

        if self.dist_map is None or float(self.dist_map.max()) == 0.0:
            path.collision_cost = 0.0
            path.safety_cost    = 0.0
            return

        pts = np.array([[s[0], s[1]] for s in path.states])
        ix = ((pts[:, 0] - self.map_origin_x) / self.map_resolution).astype(np.int32)
        iy = ((pts[:, 1] - self.map_origin_y) / self.map_resolution).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < self.map_width) & (iy >= 0) & (iy < self.map_height)

        in_bounds_count = int(np.sum(in_bounds))
        out_of_bounds_count = len(pts) - in_bounds_count
        hard_cost = 0.0
        soft_cost = 0.0

        if np.any(in_bounds):
            dist_m = self.dist_map[iy[in_bounds], ix[in_bounds]] * self.map_resolution
            hard_cost = float(np.sum(dist_m <= COLLISION_THRESHOLD))
            mask_buffer = (dist_m > COLLISION_THRESHOLD) & (dist_m < SAFETY_BUFFER)
            soft_cost = float(np.sum(SAFETY_BUFFER - dist_m[mask_buffer]))
        else:
            print(f"[ENV_COST] ⚠️  ALL {len(pts)} path points are OUT OF MAP BOUNDS! "
                  f"map=({self.map_width}x{self.map_height}), "
                  f"origin=({self.map_origin_x:.1f},{self.map_origin_y:.1f}) — "
                  f"assigning hard_cost=10.0")
            hard_cost = 10.0

        if out_of_bounds_count > 0:
            print(f"[ENV_COST] {out_of_bounds_count}/{len(pts)} path points out of map bounds")
        if hard_cost > 0:
            print(f"[ENV_COST] ⚠️  Collision: hard_cost={hard_cost:.2f} "
                  f"(pts within {COLLISION_THRESHOLD}m of obstacle)")
        if soft_cost > 0:
            print(f"[ENV_COST] Safety buffer: soft_cost={soft_cost:.2f}")

        path.collision_cost = hard_cost
        path.safety_cost = soft_cost

    def curvature_cost(self, path: PathChromosome) -> float:
        cost = 0.0
        for i in range(len(path.states) - 1):
            theta_curr = path.states[i][2]
            theta_next = path.states[i + 1][2]
            diff = (theta_next - theta_curr + math.pi) % (2 * math.pi) - math.pi
            cost += (diff ** 2)
        return cost

    def cte_cost(self, path: PathChromosome, tree, ref_points) -> float:
        if tree is None or ref_points is None or len(ref_points) == 0:
            print("[CTE_COST] ⚠️  tree or ref_points is None/empty — returning 999.0")
            return 999.0
        pts = np.array([[s[0], s[1]] for s in path.states])
        distances, _ = tree.query(pts)
        return float(np.mean(distances))

    def initialize_population(self, sx: float, sy: float, syaw: float) -> List[PathChromosome]:
        seed_yaw = syaw
        if self.ref_tree is not None and len(self.ref_points) > 0:
            snap_idx = int(getattr(self, '_last_snap_idx', 0))
            snap_idx = min(snap_idx, len(self.ref_points) - 1)

            if self.ref_points.shape[1] > 2:
                cand_yaw_check = float(self.ref_points[snap_idx, 2])
                snap_yaw_err = abs(math.atan2(
                    math.sin(syaw - cand_yaw_check),
                    math.cos(syaw - cand_yaw_check)))
                if snap_yaw_err > math.radians(45):
                    print(f"[INIT_POP] ⚠️  _last_snap_idx={snap_idx} yaw mismatch "
                          f"({math.degrees(snap_yaw_err):.1f}°) — likely stale after reverse. "
                          f"Doing global yaw-consistent re-snap ...")
                    all_yaws  = self.ref_points[:, 2]
                    all_diffs = np.abs(np.arctan2(
                        np.sin(syaw - all_yaws), np.cos(syaw - all_yaws)))
                    all_dists = np.hypot(
                        self.ref_points[:, 0] - sx,
                        self.ref_points[:, 1] - sy)
                    mask = all_diffs < math.radians(45)
                    if np.any(mask):
                        snap_idx = int(np.argmin(np.where(mask, all_dists, np.inf)))
                        self._last_snap_idx = snap_idx
                        print(f"[INIT_POP] Re-snapped after heading change → idx={snap_idx}")
                    else:
                        print(f"[INIT_POP] ⚠️  No yaw-consistent snap found within 45° — "
                              f"keeping idx={snap_idx} and using vehicle yaw as seed")

            if self.ref_points.shape[1] > 2:
                seed_yaw = float(self.ref_points[snap_idx, 2])
            else:
                next_idx = min(snap_idx + 5, len(self.ref_points) - 1)
                dx = self.ref_points[next_idx, 0] - self.ref_points[snap_idx, 0]
                dy = self.ref_points[next_idx, 1] - self.ref_points[snap_idx, 1]
                seed_yaw = math.atan2(dy, dx)

            yaw_err_ip = abs(math.atan2(math.sin(seed_yaw - syaw),
                                         math.cos(seed_yaw - syaw)))
            if yaw_err_ip > math.radians(90):
                seed_yaw = math.atan2(math.sin(seed_yaw + math.pi),
                                      math.cos(seed_yaw + math.pi))
                print(f"[INIT_POP] ⚠️  seed_yaw flipped to {math.degrees(seed_yaw):.1f}°")

            yaw_diff_check = abs(math.atan2(
                math.sin(seed_yaw - syaw), math.cos(seed_yaw - syaw)))
            if yaw_diff_check > math.radians(90):
                seed_yaw = math.atan2(math.sin(seed_yaw + math.pi),
                                      math.cos(seed_yaw + math.pi))
                print(f"[INIT_POP] ⚠️  seed_yaw was opposite direction — "
                      f"flipped to {math.degrees(seed_yaw):.1f}°")
            print(f"[INIT_POP] seed_yaw={math.degrees(seed_yaw):.1f}° "
                  f"from ref_points[{snap_idx}]  "
                  f"(vehicle yaw={math.degrees(syaw):.1f}°, "
                  f"diff={math.degrees(yaw_diff_check):.1f}°)")
        else:
            print(f"[INIT_POP] ⚠️ ref_tree not available — using vehicle yaw "
                  f"{math.degrees(syaw):.1f}° as seed")
                  
        yaw_diff = abs((seed_yaw - syaw + math.pi) % (2 * math.pi) - math.pi)
        if yaw_diff > math.radians(20):
            mutation_range = 0.6
            print(f"[INIT_POP] Large heading diff ({math.degrees(yaw_diff):.1f}°) "
                  f"— using wide mutation ±{math.degrees(mutation_range):.0f}°")
        else:
            mutation_range = 0.4
            print(f"[INIT_POP] Small heading diff ({math.degrees(yaw_diff):.1f}°) "
                  f"— using normal mutation ±{math.degrees(mutation_range):.0f}°")
                  
        print(f"[INIT_POP] Building {self.POPULATION_SIZE} chromosomes from "
              f"({sx:.3f}, {sy:.3f}) seed_yaw={math.degrees(seed_yaw):.1f}° "
              f"with {self.WAYPOINTS_PER_PATH} waypoints each ...")
              

        population: List[PathChromosome] = []
        ds = self.DELTA_S

        n_ref_elites = max(3, self.POPULATION_SIZE // 3)
        snap_idx_init = int(getattr(self, '_last_snap_idx', 0))

        if self.ref_points is not None and len(self.ref_points) > snap_idx_init + self.WAYPOINTS_PER_PATH:
            veh_to_ref_x = float(self.ref_points[snap_idx_init, 0]) - sx
            veh_to_ref_y = float(self.ref_points[snap_idx_init, 1]) - sy
            physical_cte = math.hypot(veh_to_ref_x, veh_to_ref_y)
            n_recovery_wps = min(max(int(physical_cte / 0.5) + 2, int(6 * self.V_NOMINAL / 0.3)), 20)

            for elite_i in range(n_ref_elites):
                lateral_offset = self.ga_rng.uniform(-0.2, 0.2)
                states = []
                directions = []
                for wi in range(self.WAYPOINTS_PER_PATH + 1):
                    ref_i = min(snap_idx_init + wi, len(self.ref_points) - 1)
                    rx = float(self.ref_points[ref_i, 0])
                    ry = float(self.ref_points[ref_i, 1])
                    ryaw = float(self.ref_points[ref_i, 2]) if self.ref_points.shape[1] > 2 else seed_yaw
                    if wi < n_recovery_wps and physical_cte > 0.3:
                        frac = wi / n_recovery_wps
                        px_e = sx + frac * (rx - sx)
                        py_e = sy + frac * (ry - sy)
                        blend_yaw = syaw + frac * math.atan2(
                            math.sin(ryaw - syaw), math.cos(ryaw - syaw))
                        states.append((px_e, py_e, blend_yaw))
                    else:
                        px_e = rx + lateral_offset * math.cos(ryaw + math.pi / 2)
                        py_e = ry + lateral_offset * math.sin(ryaw + math.pi / 2)
                        states.append((px_e, py_e, ryaw))
                    if wi > 0:
                        directions.append(1)
                population.append(PathChromosome(states, directions))
            print(f"[INIT_POP] Seeded {n_ref_elites} ref-elites "
                  f"(physical_cte={physical_cte:.2f}m, recovery_wps={n_recovery_wps})")
        else:
            n_ref_elites = 0
            print("[INIT_POP] ⚠️  Not enough ref_points for elite seeding — using random init only")

        if self.ref_points_raw is not None and len(self.ref_points_raw) > 0:
            _ri = min(snap_idx_init + 10, len(self.ref_points_raw) - 1)
            _ref_yaw = float(self.ref_points_raw[_ri, 2])
            _step_bias = 0.3 * math.atan2(math.sin(_ref_yaw - seed_yaw), math.cos(_ref_yaw - seed_yaw))
        else:
            _step_bias = 0.0

        n_random = self.POPULATION_SIZE - n_ref_elites
        for _ in range(n_random):
            states = [(sx, sy, seed_yaw)]
            directions = []
            curr_x, curr_y, curr_yaw = sx, sy, seed_yaw

            for _ in range(self.WAYPOINTS_PER_PATH):
                theta_t = curr_yaw + _step_bias + self.ga_rng.uniform(-mutation_range, mutation_range)
                d_t = 1
                new_x = curr_x + d_t * ds * math.cos(theta_t)
                new_y = curr_y + d_t * ds * math.sin(theta_t)
                states.append((new_x, new_y, theta_t))
                directions.append(d_t)
                curr_x, curr_y, curr_yaw = new_x, new_y, theta_t

            population.append(PathChromosome(states, directions))

        first = population[0]
        xs = [s[0] for s in first.states]
        ys = [s[1] for s in first.states]
        print(f"[INIT_POP] ✅ Done. {n_ref_elites} ref-elites + {n_random} random. "
              f"Sample chrom[0]: x=[{min(xs):.2f},{max(xs):.2f}], y=[{min(ys):.2f},{max(ys):.2f}]")
        return population

    def evaluate(self, population: List[PathChromosome], tree, ref_points) -> None:
        total_cte_sum = 0.0
        total_col_sum = 0.0
        total_fit_sum = 0.0

        for p in population:
            p.path_length = len(p.states) * self.DELTA_S * 0.05

            self.calculate_environmental_costs(p)

            pts_eval = np.array([[s[0], s[1]] for s in p.states])
            distances, _ = tree.query(pts_eval)
            p.cte_cost = float(np.mean(distances))

            smoothness_cost = 0.0
            raw_kappas = []
            for i in range(1, len(p.states)):
                angle_diff = abs(p.states[i][2] - p.states[i - 1][2])
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                smoothness_cost += abs(angle_diff)
                raw_kappas.append(abs(angle_diff) / max(self.DELTA_S, 1e-6))
            p.curvature_cost = smoothness_cost

            if len(raw_kappas) >= 3:
                smoothed_kappas = [sum(raw_kappas[max(0,i-1):i+2]) / len(raw_kappas[max(0,i-1):i+2])
                                    for i in range(len(raw_kappas))]
                max_kappa = max(smoothed_kappas)
            else:
                max_kappa = max(raw_kappas) if raw_kappas else 0.0

            curvature_violation = max(0.0, max_kappa - self.MAX_GEN_CURVATURE)
            hard_curvature_penalty = curvature_violation * 60.0

            heading_dev_cost = 0.0
            if len(p.states) > 0 and ref_points is not None and len(ref_points) > 0:
                _rp = ref_points
                for s in p.states:
                    _d, _idx = tree.query([s[0], s[1]])
                    if self.ref_points is not None:
                        _gd, _gidx = self.ref_tree.query([s[0], s[1]]) \
                            if self.ref_tree is not None else (0, 0)
                        ref_yaw = float(self.ref_points[_gidx, 2])
                        hdiff = abs(math.atan2(
                            math.sin(s[2] - ref_yaw),
                            math.cos(s[2] - ref_yaw)))
                        heading_dev_cost += hdiff
                heading_dev_cost /= max(len(p.states), 1)

            jerk_cost = 0.0
            if len(raw_kappas) >= 2:
                for i in range(1, len(raw_kappas)):
                    jerk_cost += abs(raw_kappas[i] - raw_kappas[i-1])
                jerk_cost /= max(len(raw_kappas) - 1, 1)

            total_cost = (
                p.path_length    * 1.0 +
                p.collision_cost * self.COLLISION_WEIGHT +
                p.cte_cost       * self.CTE_WEIGHT +
                p.curvature_cost * self.CURVATURE_WEIGHT +
                hard_curvature_penalty +
                heading_dev_cost * self.HEADING_DEV_WEIGHT +
                jerk_cost        * self.JERK_WEIGHT
            )
            if hasattr(self, '_boundary_tree') and self._boundary_tree is not None and len(p.states) > 0:
                pts = np.array([[s[0], s[1]] for s in p.states])
                b_dists, b_idxs = self._boundary_tree.query(pts)
                for pi in range(len(pts)):
                    if b_dists[pi] < 1.5:
                        total_cost += 20.0

            p.fitness = math.exp(-total_cost / 30.0)
            p.target_speed = self.TRAJECTORY_SPEED

            total_cte_sum += p.cte_cost
            total_col_sum += p.collision_cost
            total_fit_sum += p.fitness

        n = len(population)
        best_fit = max(p.fitness for p in population)
        print(f"[EVALUATE] n={n} | avg_CTE={total_cte_sum/n:.3f}m | "
              f"avg_collision={total_col_sum/n:.3f} | "
              f"avg_fitness={total_fit_sum/n:.5f} | best_fitness={best_fit:.5f}")

        if total_col_sum / n == 0.0:
            print("[EVALUATE] ⚠️  avg_collision=0 for ALL chromosomes — "
                  "map is empty (obs=0), collision cost is INACTIVE!")
        if best_fit < 0.001:
            print(f"[EVALUATE] ⚠️  best_fitness={best_fit:.6f} is near zero — "
                  "fitness gradient is flat, GA selection is ineffective! "
                  "Consider increasing the /100.0 denominator or reducing cost weights.")

    def generate_path_from_genes(
        self,
        genes: List[float],
        sx: float,
        sy: float,
        syaw: float,
        directions: List[int],
    ) -> PathChromosome:
        MAX_YAW_PER_STEP = self.MAX_GEN_CURVATURE * self.DELTA_S

        states = [(sx, sy, syaw)]
        curr_x, curr_y, curr_yaw = sx, sy, syaw
        for i in range(len(genes)):
            theta_t = genes[i]
            yaw_diff = math.atan2(math.sin(theta_t - curr_yaw),
                                   math.cos(theta_t - curr_yaw))
            if abs(yaw_diff) > MAX_YAW_PER_STEP:
                theta_t = curr_yaw + math.copysign(MAX_YAW_PER_STEP, yaw_diff)
            d_t = directions[i]
            new_x = curr_x + d_t * self.DELTA_S * math.cos(theta_t)
            new_y = curr_y + d_t * self.DELTA_S * math.sin(theta_t)
            states.append((new_x, new_y, theta_t))
            curr_x, curr_y, curr_yaw = new_x, new_y, theta_t
        return PathChromosome(states, directions)

    def mutate(self, parent: PathChromosome, sx: float, sy: float, syaw: float,
               mutation_mag: float = 0.1) -> PathChromosome:
        """mutation_mag controls the ±range of yaw perturbation."""
        child_genes = [s[2] for s in parent.states[1:]]
        child_directions = list(parent.directions)
        for i in range(len(child_genes)):
            if self.ga_rng.random() < self.MUTATION_RATE:
                child_genes[i] += self.ga_rng.uniform(-mutation_mag, mutation_mag)
        return self.generate_path_from_genes(child_genes, sx, sy, syaw, child_directions)

    def _republish_last_trajectory(self):
        """Re-publish last GA trajectory at 20Hz to dominate Autoware's 10Hz planner."""
        if self._goal_reached or self._emergency_stop_active or not self.autonomous_enabled:
            self.publish_stop()
            return
        if hasattr(self, '_last_best') and self._last_best is not None:
            self.publish(self._last_best)

    def run_ga(self) -> None:
        print("[RUN_GA] ✅ run_ga() entered — starting main while loop")
        self.get_logger().info("🚀 GA thread started with Kinematic Model")
        while rclpy.ok():
            if self.current_pose is None or self.ref_points is None:
                print("[RUN_GA] Waiting — pose or ref_points not ready yet ...")
                time.sleep(0.5)
                continue

            if not self.is_calibrated:
                time.sleep(0.1)
                continue

            if self._emergency_stop_active:
                if self.current_pose is not None and self.ref_tree is not None:
                    _ep_x = self.current_pose.pose.position.x
                    _ep_y = self.current_pose.pose.position.y
                    _ep_dist, _ = self.ref_tree.query([_ep_x, _ep_y])
                    _ep_clear_thresh = max(0.5, self.HARD_STOP_CTE - 1.0)
                    if float(_ep_dist) < _ep_clear_thresh:
                        print(f"[RUN_GA] ⚠️  Emergency flag set but CTE={_ep_dist:.2f}m < {_ep_clear_thresh:.1f}m "
                              f"— auto-clearing stale emergency flag")
                        self._emergency_stop_active = False
                        self._stop_anchor = None
                        self._drift_counter = 0
                    else:
                        print(f"[RUN_GA] 🛑 EMERGENCY CONFIRMED: CTE={_ep_dist:.2f}m — holding")
                        self.publish_stop()
                        time.sleep(0.2)
                        continue
                else:
                    print("[RUN_GA] 🛑 EMERGENCY STOP ACTIVE — holding evolution")
                    self.publish_stop()
                    time.sleep(0.2)
                    continue

            ga_start_time = time.perf_counter()

            with self._data_lock:
                px = self.current_pose.pose.position.x
                py = self.current_pose.pose.position.y
                car_yaw = self.current_yaw

            prev_idx = getattr(self, '_last_snap_idx', 0)
            search_start = max(0, prev_idx - 5)
            search_end = min(len(self.ref_points), prev_idx + 150)

            sub_pts = self.ref_points[search_start:search_end, :2]
            dists = (sub_pts[:, 0] - px)**2 + (sub_pts[:, 1] - py)**2
            local_min_idx = int(np.argmin(dists))
            _snap_idx = search_start + local_min_idx

            if self.ref_points.shape[1] > 2:
                cand_yaw = float(self.ref_points[_snap_idx, 2])
                yaw_err = abs(math.atan2(math.sin(car_yaw - cand_yaw), math.cos(car_yaw - cand_yaw)))
                if yaw_err > math.radians(60):
                    all_yaws = self.ref_points[:, 2]
                    all_diffs = np.abs(np.arctan2(np.sin(car_yaw - all_yaws), np.cos(car_yaw - all_yaws)))
                    all_xy_dists = np.hypot(self.ref_points[:, 0] - px, self.ref_points[:, 1] - py)
                    mask = all_diffs < math.radians(60)
                    _rg_window = 80
                    _rg_lo = max(0, prev_idx - 10)
                    _rg_hi = min(len(self.ref_points) - 1, prev_idx + _rg_window)
                    window_mask = np.zeros(len(self.ref_points), dtype=bool)
                    window_mask[_rg_lo:_rg_hi+1] = True
                    mask = mask & window_mask
                    if np.any(mask):
                        masked_dists = np.where(mask, all_xy_dists, np.inf)
                        _snap_idx = int(np.argmin(masked_dists))
                        print(f"[SNAP] ✅ Windowed yaw-consistent snap → idx={_snap_idx}")

            if int(_snap_idx) < prev_idx - 2:
                _snap_idx = prev_idx - 2
            self._last_snap_idx = int(_snap_idx)

            snap_x   = self.ref_points[int(_snap_idx), 0]
            snap_y   = self.ref_points[int(_snap_idx), 1]
            snap_yaw = self.ref_points[int(_snap_idx), 2] if self.ref_points.shape[1] > 2 else car_yaw

            current_cte = math.hypot(px - snap_x, py - snap_y)

            NEAR_WARN_M = 8.0
            FAR_WARN_M = 35.0

            def _idx_at_distance(start_i: int, target_dist: float) -> int:
                i = start_i
                acc = 0.0
                n = len(self.ref_points)
                while acc < target_dist and i < n - 1:
                    ddx = self.ref_points[i + 1][0] - self.ref_points[i][0]
                    ddy = self.ref_points[i + 1][1] - self.ref_points[i][1]
                    acc += math.hypot(ddx, ddy)
                    i += 1
                return min(i, n - 1)

            future_idx_near = _idx_at_distance(int(_snap_idx), NEAR_WARN_M)
            future_idx_far  = _idx_at_distance(int(_snap_idx), FAR_WARN_M)
            yaw_change_near = abs(math.degrees(math.atan2(
                math.sin(float(self.ref_points[future_idx_near, 2]) - snap_yaw),
                math.cos(float(self.ref_points[future_idx_near, 2]) - snap_yaw))))
            yaw_change_far  = abs(math.degrees(math.atan2(
                math.sin(float(self.ref_points[future_idx_far,  2]) - snap_yaw),
                math.cos(float(self.ref_points[future_idx_far,  2]) - snap_yaw))))
            yaw_change = max(yaw_change_near, yaw_change_far)

            if yaw_change > 60.0:
                current_lookahead = 30.0
                print(f"[RUN_GA] 🔄 Deep U-Turn! yaw_change={yaw_change:.1f}° "
                      f"Lookahead → {current_lookahead}m")
            elif yaw_change > 30.0:
                current_lookahead = 22.0
                print(f"[RUN_GA] 🔄 Moderate turn yaw_change={yaw_change:.1f}° "
                      f"Lookahead → {current_lookahead}m")
            else:
                current_lookahead = self.LOOK_AHEAD_DISTANCE

            start_idx = int(_snap_idx)
            end_idx = start_idx
            accumulated_dist = 0.0
            while accumulated_dist < current_lookahead and end_idx < len(self.ref_points) - 1:
                dx = self.ref_points[end_idx + 1][0] - self.ref_points[end_idx][0]
                dy = self.ref_points[end_idx + 1][1] - self.ref_points[end_idx][1]
                accumulated_dist += math.hypot(dx, dy)
                end_idx += 1

            min_window = 100 if yaw_change > 60.0 else (65 if yaw_change > 30.0 else 20)
            if (end_idx - start_idx) < min_window:
                end_idx = min(start_idx + min_window, len(self.ref_points) - 1)

            local_ref_points = self.ref_points[start_idx:end_idx + 1, :2]
            local_tree = KDTree(local_ref_points)

            _snap_i = int(_snap_idx)
            if _snap_i < len(self.ref_points) - 1:
                _tail = self.ref_points[_snap_i:, :2]
                _diffs = np.diff(_tail, axis=0)
                remaining_arc = float(np.sum(np.hypot(_diffs[:, 0], _diffs[:, 1])))
            else:
                remaining_arc = 0.0
            remaining_arc += math.hypot(px - snap_x, py - snap_y)

            if remaining_arc < 2.0:
                current_target_speed = 0.0
                print(f"[RUN_GA] 🏁 Near goal: remaining={remaining_arc:.2f}m → STOP")
            elif remaining_arc < 8.0:
                current_target_speed = min(0.2, self.TRAJECTORY_SPEED)
                print(f"[RUN_GA] 🐢 Approaching goal: remaining={remaining_arc:.2f}m → slow")
            elif yaw_change > 60.0:
                current_target_speed = 0.10
            elif yaw_change > 30.0:
                current_target_speed = 0.18
            else:
                current_target_speed = self.TRAJECTORY_SPEED

            if self.goal_pose is not None:
                gx = self.goal_pose.pose.position.x
                gy = self.goal_pose.pose.position.y
                dist_to_goal = math.hypot(px - gx, py - gy)
                route_progress = int(_snap_idx) / max(len(self.ref_points) - 1, 1)
                snap_exhausted = (int(_snap_idx) >= len(self.ref_points) - 3)
                # Locate the goal's own index on the reference path, so a goal
                # placed partway along the route is detected correctly. The old
                # check required the snap index to be near the END of the CSV,
                # which is only valid when the goal happens to be the last point
                # -- with a mid-route goal it never fired and the vehicle never
                # stopped.
                _goal_idx = None
                if self.ref_tree is not None:
                    try:
                        _gd, _gi = self.ref_tree.query([gx, gy])
                        _goal_idx = int(_gi)
                    except Exception:
                        _goal_idx = None
                _near_goal_idx = (_goal_idx is not None
                                  and int(_snap_idx) >= _goal_idx - 15)
                if snap_exhausted or (_near_goal_idx and dist_to_goal < 2.0):
                    current_target_speed = 0.0
                    self.autonomous_enabled = False
                    self._goal_reached = True
                    self._last_best = None
                    self._last_traj = None
                    print(f"[RUN_GA] 🏁 GOAL REACHED: dist={dist_to_goal:.2f}m "
                          f"progress={route_progress:.1%} snap_exhausted={snap_exhausted} → STOP")
                    self.publish_stop()
                    return
                elif route_progress > 0.75 and dist_to_goal < 8.0:
                    current_target_speed = min(current_target_speed, 0.15)
                elif route_progress > 0.70 and dist_to_goal < 12.0:
                    current_target_speed = min(current_target_speed, 0.20)

            self.local_tree = local_tree
            self.local_ref_points = local_ref_points

            pop = self.initialize_population(px, py, car_yaw)

            n_warm = max(3, self.POPULATION_SIZE // 3)
            warm_pop = pop[:n_warm]
            self.evaluate(warm_pop, local_tree, local_ref_points)
            warm_pop.sort(key=lambda c: c.fitness, reverse=True)
            best_warm = warm_pop[0]
            best_warm.target_speed = current_target_speed
            warm_cte = best_warm.cte_cost
            if warm_cte < 5.0:
                self._last_best = best_warm
                self.publish(best_warm)
                print(f"[RUN_GA] ⚡ Warm-start published: CTE={warm_cte:.3f}m, "
                      f"fitness={best_warm.fitness:.4f}")
            else:
                print(f"[RUN_GA] ⚠️  Warm-start skipped: CTE={warm_cte:.3f}m too high")

            _ga_cte = current_cte
            if _ga_cte > 2.0:
                _mut_mag = 0.35
            elif _ga_cte > 1.0:
                _mut_mag = 0.20
            else:
                _mut_mag = 0.10

            for gen in range(self.GENERATIONS):
                self.evaluate(pop, local_tree, local_ref_points)
                pop.sort(key=lambda c: c.fitness, reverse=True)

                n_elite = 10
                next_pop = pop[:n_elite]

                n_offspring = self.POPULATION_SIZE - n_elite
                for i in range(n_offspring):
                    parent = pop[i % n_elite]
                    child = self.mutate(parent, px, py, car_yaw, _mut_mag)
                    next_pop.append(child)
                pop = next_pop

                best_fitness = pop[0].fitness if pop else 0
                worst_fitness = pop[-1].fitness if pop else 0
                spread = best_fitness - worst_fitness
                print(f"[RUN_GA]   Gen {gen+1} after sort: "
                      f"best={best_fitness:.5f}, worst={worst_fitness:.5f}, spread={spread:.5f}")
                print(f"[RUN_GA]   Gen {gen+1} elites kept: {n_elite}")
                print(f"[RUN_GA]   Gen {gen+1} new offspring: {n_offspring}, next_gen size: {len(pop)}")

            self.evaluate(pop, local_tree, local_ref_points)
            pop.sort(key=lambda c: c.fitness, reverse=True)
            best = pop[0]
            best.target_speed = current_target_speed

            avg_cte = sum(p.cte_cost for p in pop) / len(pop) if pop else 0
            avg_col = sum(p.collision_cost for p in pop) / len(pop) if pop else 0
            avg_fit = sum(p.fitness for p in pop) / len(pop) if pop else 0
            print(f"[EVALUATE] n={len(pop)} | avg_CTE={avg_cte:.3f}m | "
                  f"avg_collision={avg_col:.3f} | avg_fitness={avg_fit:.5f} | "
                  f"best_fitness={best.fitness:.5f}")
            print(f"[RUN_GA] STEP7 Best result: fitness={best.fitness:.5f}, "
                  f"CTE={best.cte_cost:.3f}m, collision={best.collision_cost:.2f}, "
                  f"curvature={best.curvature_cost:.3f}, "
                  f"path_len={best.path_length:.2f}m, speed={current_target_speed:.2f} m/s")

            _obj_ahead = False
            if hasattr(self, "detected_objects") and self.detected_objects and self.current_pose is not None:
                _cx = float(self.current_pose.pose.position.x)
                _cy = float(self.current_pose.pose.position.y)
                _cyaw = getattr(self, "current_yaw", 0.0)
                for _ox, _oy, _or in self.detected_objects:
                    _dx = _ox - _cx
                    _dy = _oy - _cy
                    _dist = math.hypot(_dx, _dy)
                    _dot = _dx * math.cos(_cyaw) + _dy * math.sin(_cyaw)
                    if _dist < 35.0 and _dot > 0:
                        _obj_ahead = True
                        print(f"[LOOK_AHEAD] Obstacle at ({_ox:.1f},{_oy:.1f}) {_dist:.1f}m ahead — lane shift")
                        break
            _target_shift = 0.8 if _obj_ahead else 0.0
            _prev_shift = getattr(self, "_lane_shift_amount", 0.0)
            _new_shift = min(_target_shift, _prev_shift + 0.05) if _target_shift > _prev_shift else max(_target_shift, _prev_shift - 0.05)
            if abs(_new_shift - _prev_shift) > 0.001:
                _ref_orig = getattr(self, "ref_points_original", self.ref_points.copy())
                self.ref_points_original = _ref_orig
                _shifted = _ref_orig.copy()
                _yaws = _shifted[:, 2]
                _shifted[:, 0] = _ref_orig[:, 0] + _new_shift * (-np.sin(_yaws))
                _shifted[:, 1] = _ref_orig[:, 1] + _new_shift * np.cos(_yaws)
                self.ref_points = _shifted
                self.ref_tree = KDTree(self.ref_points[:, :2])
                if _new_shift > 0.01:
                    print(f"[LANE_SHIFT] ref_points shifted {_new_shift:.2f}m left")
                elif _prev_shift > 0.01:
                    print(f"[LANE_SHIFT] ref_points restored to centerline")
            self._lane_shift_amount = _new_shift
            path_len = len(best.states)
            collision_fraction = best.collision_cost / max(path_len, 1)
            if collision_fraction > 0.5:
                print(f"[RUN_GA] STEP7 ❌ {collision_fraction*100:.0f}% of path in collision "
                      f"(cost={best.collision_cost:.1f}) — publishing STOP")
                self.get_logger().error("❌ COLLISION in BEST PATH (>50% of points)")
                self.publish_stop()
            elif collision_fraction > 0.2:
                print(f"[RUN_GA] STEP7 ⚠️  {collision_fraction*100:.0f}% of path near obstacles "
                      f"— publishing with caution")
                self._last_best = best
                self.publish(best)
                self.ga_cycle += 1
            else:
                self._last_best = best
                self.publish(best)
                self.ga_cycle += 1

            ga_elapsed_ms = (time.perf_counter() - ga_start_time) * 1000
            if self.ga_cycle % 10 == 0:
                self.get_logger().info(
                    f"[RUN_GA] Cycle {self.ga_cycle} | CTE={current_cte:.3f}m | "
                    f"speed={current_target_speed:.1f}m/s | compute={ga_elapsed_ms:.1f}ms")

            time.sleep(0.02)

    def publish_rviz_alarm(self, text, color_type="warn", action="add"):
        print(f"[RVIZ_ALARM] text='{text}', type={color_type}, action={action}")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "alarms"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.DELETE if action == "delete" else Marker.ADD
        if self.current_pose:
            marker.pose.position = self.current_pose.pose.position
            marker.pose.position.z += 2.0
        marker.text = text
        marker.scale.z = 1.0
        marker.color.a = 1.0
        if color_type == "error":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        self.safe_publish(self.marker_pub, marker)

    def publish(self, best: PathChromosome) -> None:
        now = self.get_clock().now().to_msg()
        
        try:
           curvatures = []
           for i in range(len(best.states) - 1):
               dyaw = best.states[i+1][2] - best.states[i][2]
               dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
               curvatures.append(abs(dyaw) / self.DELTA_S)
           max_kappa = max(curvatures) if curvatures else 0.0
           print(f"📈 [ANALYSIS] Max Physical Curvature (kappa): {max_kappa:.3f} m^-1")
        except Exception as e:
            print(f"⚠️ Error during curvature analysis: {e}")

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = now
        for s in best.states:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(s[0])
            ps.pose.position.y = float(s[1])
            ps.pose.orientation = Quaternion(
                w=math.cos(s[2] / 2.0), x=0.0, y=0.0, z=math.sin(s[2] / 2.0)
            )
            path.poses.append(ps)
        self.safe_publish(self.path_pub, path)
        print(f"[PUBLISH] /ga_best_path: {len(path.poses)} poses published")

        traj = Trajectory()
        traj.header.frame_id = "map"
        traj.header.stamp = now
         
        raw_states = np.array(best.states)
        if len(raw_states) > 5:
            xy_points = raw_states[:, :2]
            
            window_size = 5
            smoothed_xy = np.copy(xy_points)
            for i in range(len(xy_points)):
                start = max(0, i - window_size // 2)
                end = min(len(xy_points), i + window_size // 2 + 1)
                smoothed_xy[i, 0] = np.mean(xy_points[start:end, 0])
                smoothed_xy[i, 1] = np.mean(xy_points[start:end, 1])
                
            raw_states[:, :2] = smoothed_xy
            n_s = len(smoothed_xy)
            new_yaw = raw_states[:, 2].copy()
            for i in range(n_s - 1):
                dx = smoothed_xy[i + 1, 0] - smoothed_xy[i, 0]
                dy = smoothed_xy[i + 1, 1] - smoothed_xy[i, 1]
                if math.hypot(dx, dy) > 0.01:
                    new_yaw[i] = math.atan2(dy, dx)
            if n_s > 1:
                new_yaw[-1] = new_yaw[-2]
            raw_states[:, 2] = new_yaw

        states = raw_states.tolist()
            
        for i in range(len(states) - 1):
            p1 = states[i]
            p2 = states[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            
            actual_dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            dyaw = p2[2] - p1[2]
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            
            _pub_cte = 0.0
            _pub_cte_tree = getattr(self, 'local_tree', None) or self.ref_tree
            if _pub_cte_tree is not None and self.current_pose is not None:
                try:
                    _pub_d, _ = _pub_cte_tree.query([
                        float(self.current_pose.pose.position.x),
                        float(self.current_pose.pose.position.y)])
                    _pub_cte = float(_pub_d)
                except Exception:
                    pass
            _speed_cap = (best.target_speed * 0.45) if _pub_cte > 2.0 else ((best.target_speed * 0.7) if _pub_cte > 1.0 else best.target_speed)
            dyaw = math.atan2(math.sin(p2[2] - p1[2]), math.cos(p2[2] - p1[2]))
            if abs(dyaw) > 0.10:
                dynamic_v = max(best.target_speed * 0.4, _speed_cap - (abs(dyaw) - 0.10) * 1.0)
            else:
                dynamic_v = _speed_cap

            diff = math.atan2(math.sin(p2[2] - p1[2]), math.cos(p2[2] - p1[2]))

            step_size = 0.5
            if actual_dist > step_size:
                num_mid = int(actual_dist / step_size)
                for j in range(0, num_mid):
                    frac = j / num_mid
                    tm = TrajectoryPoint()
                    tm.pose.position.x = p1[0] + frac * (p2[0] - p1[0])
                    tm.pose.position.y = p1[1] + frac * (p2[1] - p1[1])
                    m_yaw = p1[2] + frac * diff
                    tm.pose.orientation = Quaternion(w=math.cos(m_yaw/2.0), x=0.0, y=0.0, z=math.sin(m_yaw/2.0))
                    tm.longitudinal_velocity_mps = float(dynamic_v)
                    tm.front_wheel_angle_rad = 0.0
                    tm.rear_wheel_angle_rad = 0.0
                    traj.points.append(tm)
            else:
                tp = TrajectoryPoint()
                tp.pose.position.x = float(p1[0])
                tp.pose.position.y = float(p1[1])
                tp.pose.orientation = Quaternion(
                    w=math.cos(p1[2] / 2.0), x=0.0, y=0.0, z=math.sin(p1[2] / 2.0))
                tp.longitudinal_velocity_mps = float(dynamic_v)
                tp.front_wheel_angle_rad = 0.0
                tp.rear_wheel_angle_rad = 0.0
                traj.points.append(tp)
            

        if len(states) > 0:
            last_s = states[-1]
            tp_last = TrajectoryPoint()
            tp_last.pose.position.x = float(last_s[0])
            tp_last.pose.position.y = float(last_s[1])
            tp_last.pose.orientation = Quaternion(w=math.cos(last_s[2]/2.0), x=0.0, y=0.0, z=math.sin(last_s[2]/2.0))
            tp_last.longitudinal_velocity_mps = 0.0
            tp_last.front_wheel_angle_rad = 0.0
            tp_last.rear_wheel_angle_rad = 0.0
            traj.points.append(tp_last)
                
                
        if self.current_pose is not None and len(traj.points) > 0:
            ex = float(self.current_pose.pose.position.x)
            ey = float(self.current_pose.pose.position.y)
            ez = float(self.current_pose.pose.position.z)
            e_ori = self.current_pose.pose.orientation
            e_yaw = 2.0 * math.atan2(e_ori.z, e_ori.w)

            ga_first_x = traj.points[0].pose.position.x
            ga_first_y = traj.points[0].pose.position.y
            ga_first_yaw = 2.0 * math.atan2(
                traj.points[0].pose.orientation.z,
                traj.points[0].pose.orientation.w)
            gap = math.hypot(ex - ga_first_x, ey - ga_first_y)

            bear = math.atan2(ga_first_y - ey, ga_first_x - ex)
            steer_err = math.atan2(math.sin(bear - e_yaw), math.cos(bear - e_yaw))
            L = self.WHEELBASE
            recovery_steer = float(math.atan2(2.0 * L * math.sin(steer_err),
                                              max(gap, 0.5)))
            recovery_steer = max(min(recovery_steer, 0.5), -0.5)
            last_steer = getattr(self, '_last_recovery_steer', 0.0)
            max_delta = 0.4
            if abs(recovery_steer - last_steer) > max_delta:
                recovery_steer = last_steer + math.copysign(max_delta, recovery_steer - last_steer)
            self._last_recovery_steer = recovery_steer

            _arc_active = getattr(self, '_recovery_arc_active', False)
            if _arc_active:
                if gap < 0.35:
                    _arc_active = False
            else:
                if gap > 0.6:
                    _arc_active = True
            self._recovery_arc_active = _arc_active

            if _arc_active and len(traj.points) >= 2:
                n_interp = 5
                recovery_pts = []
                for k in range(1, n_interp + 1):
                    frac = k / (n_interp + 1)
                    rp = TrajectoryPoint()
                    rp.pose.position.x = ex + frac * (ga_first_x - ex)
                    rp.pose.position.y = ey + frac * (ga_first_y - ey)
                    rp.pose.position.z = ez
                    blend_yaw = e_yaw + frac * math.atan2(
                        math.sin(ga_first_yaw - e_yaw),
                        math.cos(ga_first_yaw - e_yaw))
                    rp.pose.orientation = Quaternion(
                        w=math.cos(blend_yaw / 2.0), x=0.0, y=0.0,
                        z=math.sin(blend_yaw / 2.0))
                    rp.longitudinal_velocity_mps = traj.points[0].longitudinal_velocity_mps
                    rp.front_wheel_angle_rad = 0.0
                    rp.rear_wheel_angle_rad = 0.0
                    recovery_pts.append(rp)

                ego_pt = TrajectoryPoint()
                ego_pt.pose.position.x = ex
                ego_pt.pose.position.y = ey
                ego_pt.pose.position.z = ez
                ego_pt.pose.orientation = e_ori
                ego_pt.longitudinal_velocity_mps = traj.points[0].longitudinal_velocity_mps
                ego_pt.front_wheel_angle_rad = 0.0
                ego_pt.rear_wheel_angle_rad = 0.0
                for rp in reversed(recovery_pts):
                    traj.points.insert(0, rp)
                traj.points.insert(0, ego_pt)
                print(f"[PUBLISH] Recovery arc: gap={gap:.2f}m steer={math.degrees(recovery_steer):.1f}°")
            else:
                ego_pt = TrajectoryPoint()
                ego_pt.pose.position.x = ex
                ego_pt.pose.position.y = ey
                ego_pt.pose.position.z = ez
                ego_pt.pose.orientation = e_ori
                ego_pt.longitudinal_velocity_mps = traj.points[0].longitudinal_velocity_mps
                ego_pt.front_wheel_angle_rad = 0.0
                ego_pt.rear_wheel_angle_rad = 0.0
                traj.points.insert(0, ego_pt)

        if len(traj.points) >= 2 and self.current_pose is not None:
            e_yaw_pp = 2.0 * math.atan2(
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w)
            ego_px = float(self.current_pose.pose.position.x)
            ego_py = float(self.current_pose.pose.position.y)
            cos_h = math.cos(e_yaw_pp)
            sin_h = math.sin(e_yaw_pp)

            filtered = [traj.points[0]]
            for pt in traj.points[1:]:
                dx = pt.pose.position.x - ego_px
                dy = pt.pose.position.y - ego_py
                if dx * cos_h + dy * sin_h > -0.3:
                    filtered.append(pt)
            traj.points = filtered

        self.safe_publish(self.traj_pub, traj)
        print(f"[PUBLISH] /trajectory: {len(traj.points)} points published (ego prepended)")

    def clear_rviz(self):
        print("[CLEAR_RVIZ] Clearing all published visuals")
        try:
            if rclpy.ok():
                self.safe_publish(self.path_pub, Path(header=Header(frame_id="map")))
                empty_traj = Trajectory(header=Header(frame_id="map"))
                empty_traj.header.stamp = self.get_clock().now().to_msg()
                self.safe_publish(self.traj_pub, empty_traj)
                self.publish_rviz_alarm("", action="delete")
        except Exception as e:
            print(f"[CLEAR_RVIZ] Skipped — node shutting down: {e}")
        self._last_best = None
        print("[CLEAR_RVIZ] Done")

    def compute_target_speed(self, snap_idx: int) -> float:
        """Target speed from upcoming curvature: v = clip(sqrt(A_LAT_MAX/kappa), V_MIN, V_NOMINAL)."""
        if self.ref_points is None or len(self.ref_points) < 2:
            return self.V_NOMINAL

        n = len(self.ref_points)
        i = int(snap_idx)
        acc_dist = 0.0
        max_kappa = 0.0

        while acc_dist < self.V_PLAN_HORIZON and i < n - 1:
            dx = float(self.ref_points[i + 1, 0]) - float(self.ref_points[i, 0])
            dy = float(self.ref_points[i + 1, 1]) - float(self.ref_points[i, 1])
            ds = math.hypot(dx, dy)
            if ds < 1e-6:
                i += 1
                continue
            dyaw = float(self.ref_points[i + 1, 2]) - float(self.ref_points[i, 2])
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
            kappa = abs(dyaw) / ds
            max_kappa = max(max_kappa, kappa)
            acc_dist += ds
            i += 1

        if max_kappa > 1e-4:
            v_curve = math.sqrt(self.A_LAT_MAX / max_kappa)
        else:
            v_curve = self.V_NOMINAL

        v_target = float(np.clip(v_curve, self.V_MIN, self.V_NOMINAL))
        return v_target

    def publish_control(self):
        if not self.autonomous_enabled:
            return
        if self._last_best is None or self.current_pose is None:
            return
        try:
            g_cmd = GearCommand()
            g_cmd.stamp = self.get_clock().now().to_msg()
            if self._last_best.directions:
                g_cmd.command = GearCommand.DRIVE if self._last_best.directions[0] > 0 else 4
            else:
                g_cmd.command = GearCommand.DRIVE
            self.safe_publish(self.gear_pub, g_cmd)

            ctrl = Control()
            ctrl.stamp = self.get_clock().now().to_msg()
            ctrl.longitudinal.acceleration = 0.5

            default_lookahead = self.LOOK_AHEAD_DISTANCE
            target_state = None
            best_diff = float('inf')
            _cx = self.current_pose.pose.position.x
            _cy = self.current_pose.pose.position.y
            _cyaw = getattr(self, 'current_yaw', 0.0)
            for state in self._last_best.states:
                dx = state[0] - _cx
                dy = state[1] - _cy
                if dx * math.cos(_cyaw) + dy * math.sin(_cyaw) < 0.3:
                    continue
                dist = math.hypot(dx, dy)
                diff = abs(dist - default_lookahead)
                if diff < best_diff:
                    best_diff = diff
                    target_state = state

            if target_state is None:
                target_state = self._last_best.states[min(15, len(self._last_best.states) - 1)]

            dx = target_state[0] - self.current_pose.pose.position.x
            dy = target_state[1] - self.current_pose.pose.position.y
            bearing_to_target = math.atan2(dy, dx)
            current_yaw = getattr(self, 'current_yaw', 0.0)
            error = bearing_to_target - current_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            yaw_err_abs = abs(math.degrees(error))
            if yaw_err_abs > 60.0:
                lookahead_dist = 5.0
            elif yaw_err_abs > 40.0:
                lookahead_dist = 8.0
            else:
                lookahead_dist = self.LOOK_AHEAD_DISTANCE

            if lookahead_dist != default_lookahead:
                target_state2 = None
                best_diff2 = float('inf')
                for state in self._last_best.states:
                    dx2 = state[0] - _cx
                    dy2 = state[1] - _cy
                    if dx2 * math.cos(_cyaw) + dy2 * math.sin(_cyaw) < 0.3:
                        continue
                    dist2 = math.hypot(dx2, dy2)
                    diff2 = abs(dist2 - lookahead_dist)
                    if diff2 < best_diff2:
                        best_diff2 = diff2
                        target_state2 = state
                if target_state2 is not None:
                    target_state = target_state2
                    dx = target_state[0] - _cx
                    dy = target_state[1] - _cy
                    bearing_to_target = math.atan2(dy, dx)
                    error = bearing_to_target - current_yaw
                    error = math.atan2(math.sin(error), math.cos(error))
                    yaw_err_abs = abs(math.degrees(error))

            K_MIN, K_MAX = 0.18, 0.35
            K_RAMP_START_DEG, K_RAMP_END_DEG = 0.0, 60.0
            _ramp_t = max(0.0, min(1.0, (yaw_err_abs - K_RAMP_START_DEG)
                                   / (K_RAMP_END_DEG - K_RAMP_START_DEG)))
            K = K_MIN + _ramp_t * (K_MAX - K_MIN)
            raw_steer = max(min(float(error * K), 0.4), -0.4)

            physical_cte_now = 0.0
            _cte_tree = getattr(self, 'local_tree', None)
            if _cte_tree is None:
                _cte_tree = self.ref_tree
            if _cte_tree is not None and self.current_pose is not None:
                try:
                    _d, _ = _cte_tree.query([_cx, _cy])
                    physical_cte_now = float(_d)
                except Exception:
                    pass

            if not hasattr(self, '_large_yaw_err_count'):
                self._large_yaw_err_count = 0
            target_v = self.TRAJECTORY_SPEED

            _enter_mild = self.CTE_ENTER_MILD
            _enter_recovery = self.CTE_ENTER_RECOVERY

            prev_mode = self.CTE_MODE
            if prev_mode == "NORMAL":
                if physical_cte_now > _enter_mild:
                    self.CTE_MODE = "MILD"
            elif prev_mode == "MILD":
                if physical_cte_now > _enter_recovery:
                    self.CTE_MODE = "RECOVERY"
                elif physical_cte_now < self.CTE_EXIT_MILD:
                    self.CTE_MODE = "NORMAL"
            elif prev_mode == "RECOVERY":
                if physical_cte_now < self.CTE_EXIT_RECOVERY:
                    self.CTE_MODE = "MILD"
            if self.CTE_MODE != prev_mode:
                print(f"[CTRL] 🔁 mode {prev_mode} → {self.CTE_MODE} "
                      f"(CTE={physical_cte_now:.2f}m)")

            if self.CTE_MODE == "RECOVERY" and self.ref_points is not None:
                LOOKAHEAD_M = 4.0
                best_fwd_idx = None
                best_fwd_dist_diff = float('inf')
                snap_i = int(getattr(self, '_last_snap_idx', 0))
                search_start = max(0, snap_i - 10)
                search_end = min(len(self.ref_points), snap_i + 80)
                for ri in range(search_start, search_end):
                    rpx = self.ref_points[ri, 0]
                    rpy = self.ref_points[ri, 1]
                    rdx = rpx - _cx
                    rdy = rpy - _cy
                    if rdx * math.cos(_cyaw) + rdy * math.sin(_cyaw) < 0.1:
                        continue
                    dist_ri = math.hypot(rdx, rdy)
                    diff = abs(dist_ri - LOOKAHEAD_M)
                    if diff < best_fwd_dist_diff:
                        best_fwd_dist_diff = diff
                        best_fwd_idx = ri

                if best_fwd_idx is not None:
                    tgt_x = float(self.ref_points[best_fwd_idx, 0])
                    tgt_y = float(self.ref_points[best_fwd_idx, 1])
                    ld = math.hypot(tgt_x - _cx, tgt_y - _cy)
                    bear_fwd = math.atan2(tgt_y - _cy, tgt_x - _cx)
                    alpha_fwd = math.atan2(
                        math.sin(bear_fwd - _cyaw),
                        math.cos(bear_fwd - _cyaw))
                    L = 2.79
                    recovery_steer = math.atan2(
                        2.0 * L * math.sin(alpha_fwd), max(ld, 0.5))
                    recovery_steer = max(min(recovery_steer, 0.6), -0.6)
                    desired_steer = recovery_steer
                    target_v = 0.15
                    if self._ctrl_print_counter % 25 == 0:
                        print(f"[CTRL] 🚨 RECOVERY: CTE={physical_cte_now:.2f}m "
                              f"target=ref[{best_fwd_idx}] "
                              f"alpha={math.degrees(alpha_fwd):.1f}° "
                              f"steer={math.degrees(recovery_steer):.1f}°")
                else:
                    desired_steer = raw_steer
                    target_v = 0.0

            elif self.CTE_MODE == "MILD":
                desired_steer = raw_steer
                yaw_err_deg = abs(math.degrees(error))
                target_v = 0.20 if yaw_err_deg < 60.0 else 0.15
                if self._ctrl_print_counter % 50 == 0:
                    print(f"[CTRL] ⚠️  CTE={physical_cte_now:.2f}m — correction steer={math.degrees(raw_steer):.1f}°")

            else:
                desired_steer = raw_steer
                yaw_err_deg = abs(math.degrees(error))
                ts = self.TRAJECTORY_SPEED
                if yaw_err_deg > 90.0:
                    target_v = min(0.15, ts * 0.35)
                elif yaw_err_deg > 60.0:
                    target_v = min(0.20, ts * 0.50)
                elif yaw_err_deg > 40.0:
                    target_v = min(0.28, ts * 0.70)
                else:
                    target_v = ts

            if self.CTE_MODE == "NORMAL":
                _slew = self.STEER_SLEW_RATE_MAX
            elif self.CTE_MODE == "MILD":
                _slew = self.STEER_SLEW_RATE_MAX * 4.0
            else:
                _slew = self.STEER_SLEW_RATE_MAX * 8.0
            _delta = desired_steer - self._prev_steering_tire_angle
            _delta = max(min(_delta, _slew), -_slew)
            ctrl.lateral.steering_tire_angle = self._prev_steering_tire_angle + _delta
            self._prev_steering_tire_angle = ctrl.lateral.steering_tire_angle

            ctrl.longitudinal.velocity = target_v
            self.safe_publish(self.control_pub, ctrl)

            if not hasattr(self, '_ctrl_print_counter'):
                self._ctrl_print_counter = 0
            self._ctrl_print_counter += 1
            if self._ctrl_print_counter % 50 == 0:
                print(f"[CTRL] vel={target_v:.2f} m/s | "
                      f"steer={math.degrees(ctrl.lateral.steering_tire_angle):.2f}° "
                      f"(mode={self.CTE_MODE}, K={K:.2f}) | "
                      f"yaw_err={math.degrees(error):.2f}° | "
                      f"gear={'Drive' if g_cmd.command == 2 else 'Reverse'}")

        except Exception as e:
            print(f"[CTRL] ❌ Exception in publish_control: {e}")
            self.get_logger().error(f"Control Timer Error: {e}")
    def publish_stop(self) -> None:
        print("[STOP] 🛑 publish_stop() called")
        now = self.get_clock().now().to_msg()

        traj = Trajectory()
        traj.header.frame_id = "map"
        traj.header.stamp = now
        # Anchor the stop trajectory at a FIXED pose captured on the first
        # stop call, not at the live current_pose. publish_stop() runs at 50Hz;
        # rebuilding a forward-extending ramp from the live pose every cycle
        # means the zero-velocity end of the ramp keeps moving away and the
        # vehicle chases the ramp's non-zero head forever instead of stopping.
        if getattr(self, '_stop_anchor', None) is None:
            if self.current_pose is None:
                return
            self._stop_anchor = self.current_pose
        anchor = self._stop_anchor
        _a_yaw = 2.0 * math.atan2(anchor.pose.orientation.z,
                                  anchor.pose.orientation.w)
        for k in range(5):
            sp = TrajectoryPoint()
            # Space points forward along the anchor heading. They must NOT be
            # coincident: the MPC derives path heading from the vector between
            # consecutive points, and zero-length segments give it no usable
            # direction, which produces garbage steering. Geometry stays valid
            # and straight; velocity is zero everywhere, so the vehicle stops
            # without turning.
            sp.pose.position.x = anchor.pose.position.x + k * 0.5 * math.cos(_a_yaw)
            sp.pose.position.y = anchor.pose.position.y + k * 0.5 * math.sin(_a_yaw)
            sp.pose.position.z = anchor.pose.position.z
            sp.pose.orientation = anchor.pose.orientation
            sp.longitudinal_velocity_mps = 0.0
            sp.acceleration_mps2 = 0.0
            sp.front_wheel_angle_rad = 0.0
            sp.rear_wheel_angle_rad = 0.0
            traj.points.append(sp)
        self.safe_publish(self.traj_pub, traj)
        print("[STOP] Zero-velocity stop trajectory published (anchored)")

def main():
    print("[MAIN] rclpy.init() ...")
    rclpy.init()
    print("[MAIN] Creating GA_PlannerNode ...")
    node = GA_PlannerNode()
    print("[MAIN] Node created — entering rclpy.spin()")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt received — shutting down")
    finally:
        print("[MAIN] Destroying node ...")
        node.destroy_node()
        rclpy.shutdown()
        print("[MAIN] Clean shutdown complete")

if __name__ == "__main__":
    main()
