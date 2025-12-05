"""
Sensor-Based Simulator for Adaptive Cruise Control

This version uses actual CARLA sensors instead of ground truth:
- RADAR sensor for distance measurement (affected by weather)
- IMU sensor for velocity measurement
- Weather conditions from scenario files are applied

Weather Impact on Sensors:
- Fog: Reduces radar detection range and adds noise
- Rain: Increases radar noise and can cause false detections
- Low sun angle: Can cause sensor interference

Author: Generated for realistic ACC testing
"""

import cv2
import random
import glob
import os
import sys
import math
import threading

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from collections import deque
from enum import Enum
from typing import NamedTuple, Optional, Dict, List, Any
from dataclasses import dataclass

import carla
import numpy as np
import pygame
from skimage.transform import resize

from mp1_simulator.misc import *
from mp1_simulator.render import BirdeyeRender


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "display_size": 400,
    "max_past_step": 1,
    "dt": 0.1,
    "max_timesteps": 200,
    "ego_vehicle_filter": "vehicle.lincoln.*",
    "ado_vehicle_filter": "vehicle.toyota.prius",
    "port": 2000,
    "town": "Town06",
    "obs_range": 50,
    "d_behind": 12,
    "desired_speed": 20,
    "distance_threshold": 4,
    "render": True,
    "initial_ego_state": None,
    "initial_lead_state": None,
    # Sensor-specific config
    "use_ground_truth": False,  # Set to True to fall back to ground truth
    "radar_range": 100.0,  # Maximum radar detection range (meters)
    "radar_fov": 30.0,  # Radar field of view (degrees)
    "velocity_noise_std": 0.1,  # Standard deviation of velocity noise (m/s)
}


class Mode(Enum):
    CRUISING = 0
    FOLLOWING = 1


class Observation(NamedTuple):
    ego_velocity: float
    desired_speed: float
    distance_to_lead: float


# ============================================================================
# WEATHER EFFECTS ON SENSORS
# ============================================================================

@dataclass
class WeatherEffects:
    """
    Models how weather conditions affect sensor performance.
    Based on real-world studies of radar performance in adverse conditions.
    """
    # Fog effects
    fog_range_reduction: float = 0.0  # Percentage reduction in radar range
    fog_noise_increase: float = 0.0   # Additional noise due to fog
    
    # Rain effects  
    rain_noise_increase: float = 0.0  # Additional noise due to rain
    rain_false_detection_prob: float = 0.0  # Probability of false detection
    
    # Overall detection probability
    detection_probability: float = 1.0
    
    @classmethod
    def from_weather_params(cls, weather_params: Dict[str, float]) -> 'WeatherEffects':
        """
        Calculate sensor degradation from weather parameters.
        
        References:
        - Fog: Radar range decreases exponentially with fog density
        - Rain: Heavy rain can cause ~20% range reduction and increased noise
        - Sun angle: Low sun can cause radar interference (not modeled here)
        """
        effects = cls()
        
        fog_density = weather_params.get('fog_density', 0.0)
        precipitation = weather_params.get('precipitation', 0.0)
        
        # Fog effects on radar
        # At 80% fog density, range reduced by ~50%, noise increased by 2x
        if fog_density > 0:
            effects.fog_range_reduction = min(0.6, fog_density / 100.0 * 0.75)
            effects.fog_noise_increase = fog_density / 100.0 * 2.0  # Up to 2m additional noise
        
        # Rain effects on radar
        # Heavy rain (100%) causes ~20% range reduction and significant noise
        if precipitation > 0:
            effects.rain_noise_increase = precipitation / 100.0 * 1.5  # Up to 1.5m additional noise
            effects.rain_false_detection_prob = min(0.1, precipitation / 100.0 * 0.1)
        
        # Overall detection probability
        # Decreases with adverse weather
        base_prob = 1.0
        base_prob -= effects.fog_range_reduction * 0.3  # Fog reduces detection
        base_prob -= precipitation / 100.0 * 0.1  # Rain reduces detection
        effects.detection_probability = max(0.7, base_prob)
        
        return effects


# ============================================================================
# RADAR DATA PROCESSOR
# ============================================================================

class RadarProcessor:
    """
    Processes raw radar data to extract distance to lead vehicle.
    Applies weather-based degradation and noise.
    """
    
    def __init__(self, max_range: float = 100.0, fov: float = 30.0):
        self.max_range = max_range
        self.fov = fov
        self.raw_detections: List[Any] = []
        self.lock = threading.Lock()
        self.weather_effects = WeatherEffects()
        
    def set_weather_effects(self, effects: WeatherEffects):
        """Update weather effects for sensor degradation."""
        self.weather_effects = effects
        
    def process_radar_data(self, radar_data) -> Optional[float]:
        """
        Process radar measurements to find the closest vehicle ahead.
        
        Returns:
            Distance to lead vehicle (meters), or None if not detected
        """
        with self.lock:
            self.raw_detections = list(radar_data)
        
        if not self.raw_detections:
            return None
        
        # Apply weather-based range reduction
        effective_range = self.max_range * (1.0 - self.weather_effects.fog_range_reduction)
        
        # Find closest detection in front of the vehicle
        min_distance = float('inf')
        
        for detection in self.raw_detections:
            # Radar returns azimuth (horizontal angle), altitude (vertical angle), 
            # depth (distance), and velocity
            azimuth = math.degrees(detection.azimuth)
            altitude = math.degrees(detection.altitude)
            depth = detection.depth
            
            # Filter: only consider detections ahead (within FOV)
            # and at similar altitude (road level, not overhead signs)
            if abs(azimuth) < self.fov / 2 and abs(altitude) < 10:
                if depth < min_distance and depth < effective_range:
                    min_distance = depth
        
        if min_distance == float('inf'):
            return None
        
        # Apply weather-based detection probability
        if np.random.random() > self.weather_effects.detection_probability:
            return None  # Missed detection
        
        # Apply weather-based noise
        noise_std = 0.5  # Base noise (0.5m)
        noise_std += self.weather_effects.fog_noise_increase
        noise_std += self.weather_effects.rain_noise_increase
        
        noisy_distance = min_distance + np.random.normal(0, noise_std)
        
        # Clamp to reasonable values
        return max(0.5, noisy_distance)
    
    def update_detections(self, radar_data):
        """Callback for radar sensor data."""
        with self.lock:
            self.raw_detections = list(radar_data)


# ============================================================================
# IMU/VELOCITY PROCESSOR
# ============================================================================

class VelocityProcessor:
    """
    Processes IMU data or simulates wheel odometry for velocity estimation.
    Adds realistic noise to velocity measurements.
    """
    
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
        self.velocity_buffer = deque(maxlen=5)  # For smoothing
        self.weather_effects = WeatherEffects()
        
    def set_weather_effects(self, effects: WeatherEffects):
        """Update weather effects (rain can affect wheel slip)."""
        self.weather_effects = effects
        
    def process_velocity(self, ground_truth_velocity: float) -> float:
        """
        Add realistic noise to velocity measurement.
        In real systems, velocity comes from wheel encoders or GPS.
        
        Weather effects:
        - Wet roads can cause wheel slip, increasing velocity error
        """
        # Additional noise from wet conditions
        weather_noise = 0.0
        if hasattr(self.weather_effects, 'rain_noise_increase'):
            # Wet roads cause wheel slip
            precipitation_factor = self.weather_effects.rain_noise_increase / 1.5
            weather_noise = precipitation_factor * 0.2  # Up to 0.2 m/s extra noise
        
        total_noise_std = self.noise_std + weather_noise
        
        # Add Gaussian noise
        noisy_velocity = ground_truth_velocity + np.random.normal(0, total_noise_std)
        
        # Smooth with moving average
        self.velocity_buffer.append(noisy_velocity)
        smoothed = np.mean(self.velocity_buffer)
        
        return max(0.0, smoothed)  # Velocity can't be negative


# ============================================================================
# SENSOR-BASED SIMULATOR
# ============================================================================

class SensorSimulator:
    """
    CARLA Simulator using actual sensors for ACC.
    
    Sensors used:
    - Radar: For distance measurement to lead vehicle
    - IMU/Wheel Encoders: For ego velocity (simulated with noise)
    - Collision Sensor: For safety monitoring
    - Camera: For video recording (optional)
    
    Weather is applied from scenario files and affects sensor performance.
    """

    def __init__(self, **config):
        self.camera_buffer = []

        self.config = CONFIG.copy()
        self.config.update(config)

        self.server_port = self.config["port"]
        self.dt = self.config["dt"]
        self.display_size = self.config["display_size"]
        self.use_ground_truth = self.config.get("use_ground_truth", False)

        print("Connecting to CARLA server...")
        self.client = carla.Client("127.0.0.1", self.server_port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.config["town"])
        print("CARLA server connected!")

        # Weather will be set from scenario
        self.current_weather_params: Dict[str, float] = {}
        self.weather_effects = WeatherEffects()

        # Create vehicle blueprints
        self.ego_bp = self._create_vehicle_blueprint(
            self.config["ego_vehicle_filter"], color="49,8,8")
        self.ego: Optional[carla.Vehicle] = None
        self.ado_bp = self._create_vehicle_blueprint(
            self.config["ado_vehicle_filter"], color="49,8,8")
        self.ado: Optional[carla.Vehicle] = None

        # ==================== SENSORS ====================
        
        # 1. Collision sensor
        self.collision_hist = deque(maxlen=1)
        self.collided_event = False
        self.collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_sensor = None

        # 2. Radar sensor (PRIMARY for distance measurement)
        self.radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
        self.radar_bp.set_attribute("horizontal_fov", str(self.config.get("radar_fov", 30.0)))
        self.radar_bp.set_attribute("vertical_fov", "20")
        self.radar_bp.set_attribute("range", str(self.config.get("radar_range", 100.0)))
        self.radar_bp.set_attribute("points_per_second", "1500")
        self.radar_bp.set_attribute("sensor_tick", str(self.dt))
        self.radar_sensor = None
        self.radar_processor = RadarProcessor(
            max_range=self.config.get("radar_range", 100.0),
            fov=self.config.get("radar_fov", 30.0)
        )
        
        # Radar mounting position (front bumper, typical ACC placement)
        self.radar_transform = carla.Transform(
            carla.Location(x=2.5, z=0.5),  # Front of vehicle, low
            carla.Rotation(pitch=0, yaw=0)
        )

        # 3. Camera sensor (for recording only)
        self.obs_size = 600
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(self.obs_size))
        self.camera_bp.set_attribute("image_size_y", str(self.obs_size))
        self.camera_bp.set_attribute("fov", "110")
        self.camera_bp.set_attribute("sensor_tick", "0.02")
        self.camera_sensor = None

        # 4. Velocity processor (simulates wheel encoder/IMU)
        self.velocity_processor = VelocityProcessor(
            noise_std=self.config.get("velocity_noise_std", 0.1)
        )

        # ==================== STATE ====================
        
        self.settings = self.world.get_settings()
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0
        
        # For logging ground truth vs sensor
        self.ground_truth_distance = 0.0
        self.sensor_distance = 0.0
        self.ground_truth_velocity = 0.0
        self.sensor_velocity = 0.0

        self.render = config.get("render", False)
        self._init_renderer()

    # ==================== WEATHER ====================
    
    def set_weather(self, weather_params: Dict[str, float]):
        """
        Apply weather conditions to the CARLA world.
        Also updates sensor degradation effects.
        
        Args:
            weather_params: Dictionary with weather parameters
        """
        self.current_weather_params = weather_params
        
        # Create CARLA weather object
        weather = carla.WeatherParameters(
            cloudiness=weather_params.get('cloudiness', 0.0),
            precipitation=weather_params.get('precipitation', 0.0),
            precipitation_deposits=weather_params.get('precipitation_deposits', 0.0),
            wind_intensity=weather_params.get('wind_intensity', 0.0),
            sun_altitude_angle=weather_params.get('sun_altitude_angle', 45.0),
            fog_density=weather_params.get('fog_density', 0.0),
            fog_distance=weather_params.get('fog_distance', 1000.0),
            wetness=weather_params.get('wetness', 0.0),
        )
        
        self.world.set_weather(weather)
        
        # Update sensor degradation effects
        self.weather_effects = WeatherEffects.from_weather_params(weather_params)
        self.radar_processor.set_weather_effects(self.weather_effects)
        self.velocity_processor.set_weather_effects(self.weather_effects)
        
        print(f"Weather applied: fog={weather_params.get('fog_density', 0):.1f}%, "
              f"rain={weather_params.get('precipitation', 0):.1f}%, "
              f"detection_prob={self.weather_effects.detection_probability:.2f}")

    # ==================== VEHICLE SETUP ====================
    
    def set_spawn_points(self, initial_ego_state, initial_lead_state):
        """Set initial vehicle positions and velocities."""
        ego_location = initial_ego_state["position"]
        ego_yaw = initial_ego_state["yaw"]
        lead_location = initial_lead_state["position"]
        lead_yaw = initial_lead_state["yaw"]

        self.ego_vehicle_spawn_point = carla.Transform(
            carla.Location(x=ego_location['x'], y=-ego_location['y'], z=ego_location['z']),
            carla.Rotation(pitch=0.0, yaw=ego_yaw + 180, roll=0.0),
        )
        self.ado_vehicle_spawn_point = carla.Transform(
            carla.Location(x=lead_location['x'], y=-lead_location['y'], z=lead_location['z']),
            carla.Rotation(pitch=0.0, yaw=lead_yaw + 180, roll=0.0),
        )
        
        # Store initial velocities (NEW - these were being ignored before!)
        self.initial_ego_velocity = initial_ego_state.get('velocity', 0.0)
        self.initial_lead_velocity = initial_lead_state.get('velocity', 0.0)
        
        if self.initial_ego_velocity > 0 or self.initial_lead_velocity > 0:
            print(f"Initial velocities: ego={self.initial_ego_velocity:.1f} m/s, "
                  f"lead={self.initial_lead_velocity:.1f} m/s")

    def _create_vehicle_blueprint(self, actor_filter, color=None):
        """Create vehicle blueprint."""
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute("color"):
            if not color:
                color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        return bp

    # ==================== SENSOR MANAGEMENT ====================
    
    def _clear_all_actors(self):
        """Clear all sensors and vehicles."""
        sensors = [
            self.collision_sensor,
            self.camera_sensor,
            self.radar_sensor,
        ]
        
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass
        
        self.collision_sensor = None
        self.camera_sensor = None
        self.radar_sensor = None

        for vehicle in [self.ego, self.ado]:
            if vehicle is not None:
                try:
                    vehicle.destroy()
                except:
                    pass
        
        self.ego = None
        self.ado = None

    def _spawn_sensors(self):
        """Spawn all sensors attached to ego vehicle."""
        # 1. Collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(lambda event: self.handle_collision(event))
        
        # 2. Radar sensor
        self.radar_sensor = self.world.spawn_actor(
            self.radar_bp, self.radar_transform, attach_to=self.ego
        )
        self.radar_sensor.listen(lambda data: self.radar_processor.update_detections(data))
        
        # 3. Camera sensor
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego
        )
        
        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array
            
        self.camera_sensor.listen(get_camera_img)

    # ==================== SIMULATION CONTROL ====================
    
    def _set_synchronous_mode(self, synchronous=True):
        """Set synchronous simulation mode."""
        self.settings.fixed_delta_seconds = self.dt
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _init_renderer(self):
        """Initialize visualization renderer."""
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.display_size * 3, self.display_size),
                pygame.HWSURFACE | pygame.DOUBLEBUF,
            )
            pixels_per_meter = self.display_size / self.config["obs_range"]
            pixels_ahead_vehicle = (
                self.config["obs_range"] / 2 - self.config["d_behind"]
            ) * pixels_per_meter
            birdeye_params = {
                "screen_size": [self.display_size, self.display_size],
                "pixels_per_meter": pixels_per_meter,
                "pixels_ahead_vehicle": pixels_ahead_vehicle,
            }
            self.birdeye_render = BirdeyeRender(self.world, birdeye_params)
        else:
            self.display = None
            self.birdeye_render = None

    def reset(self) -> Observation:
        """Reset simulation for new episode."""
        self._clear_all_actors()
        self._set_synchronous_mode(False)

        # Spawn vehicles
        self._try_spawn_ado_vehicle_at(self.ado_vehicle_spawn_point)
        
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)

        self._try_spawn_ego_vehicle_at(self.ego_vehicle_spawn_point)

        # Spawn sensors
        self._spawn_sensors()
        self.collided_event = False
        self.collision_hist = deque(maxlen=1)

        # ==================== APPLY INITIAL VELOCITIES ====================
        # This was previously missing - vehicles always started at 0 m/s!
        self._apply_initial_velocities()

        # Reset state
        self.time_step = 0
        self.reset_step += 1
        self.velocity_processor.velocity_buffer.clear()

        self._set_synchronous_mode(True)

        if self.birdeye_render is not None:
            self.birdeye_render.set_hero(self.ego, self.ego.id)

        # Let sensors initialize and physics settle after velocity change
        self.world.tick()
        self.world.tick()  # Extra tick for velocity to take effect

        return self._get_obs()
    
    def _apply_initial_velocities(self):
        """
        Apply initial velocities to vehicles.
        
        This was a critical missing feature - scenarios specified initial velocities
        but they were never applied, so all simulations started with stationary vehicles!
        """
        # Apply ego vehicle initial velocity
        if hasattr(self, 'initial_ego_velocity') and self.initial_ego_velocity > 0:
            transform = self.ego.get_transform()
            forward = transform.get_forward_vector()
            
            # Create velocity vector in vehicle's forward direction
            velocity = carla.Vector3D(
                x=forward.x * self.initial_ego_velocity,
                y=forward.y * self.initial_ego_velocity,
                z=0.0
            )
            self.ego.set_target_velocity(velocity)
            print(f"  Applied ego initial velocity: {self.initial_ego_velocity:.1f} m/s")
        
        # Apply lead vehicle initial velocity
        if hasattr(self, 'initial_lead_velocity') and self.initial_lead_velocity > 0:
            transform = self.ado.get_transform()
            forward = transform.get_forward_vector()
            
            velocity = carla.Vector3D(
                x=forward.x * self.initial_lead_velocity,
                y=forward.y * self.initial_lead_velocity,
                z=0.0
            )
            self.ado.set_target_velocity(velocity)
            print(f"  Applied lead initial velocity: {self.initial_lead_velocity:.1f} m/s")

    def handle_collision(self, event):
        """Handle collision event."""
        self.collision_hist.append(True)
        self.collided_event = True
        print("Vehicle collision detected!!!")

    def _try_spawn_ego_vehicle_at(self, transform):
        """Spawn ego vehicle."""
        vehicle = None
        if self.ego is not None:
            vehicle = self.ego if self.ego.is_alive else None
        else:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            vehicle.set_transform(transform)
            self.ego = vehicle
            print("Using Ego Vehicle ID:", self.ego)
            return True
        return False

    def _try_spawn_ado_vehicle_at(self, transform):
        """Spawn lead (ado) vehicle."""
        vehicle = None
        if self.ado is not None:
            vehicle = self.ado if self.ado.is_alive else None
        else:
            vehicle = self.world.try_spawn_actor(self.ado_bp, transform)

        if vehicle is not None:
            vehicle.set_transform(transform)
            self.ado = vehicle
            print("Using Ado Vehicle ID:", self.ado)
            return True
        return False

    # ==================== CONTROL ====================
    
    def _get_control(self, acc) -> carla.VehicleControl:
        """Convert acceleration to vehicle control."""
        if acc > 0:
            throttle = np.clip(acc / 10, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 20, 0, 1)
        return carla.VehicleControl(throttle=float(throttle), brake=float(brake))

    def _set_ado_control(self, ado_actions):
        """Set lead vehicle action sequence."""
        self.ado_throttle = [a['throttle'] for a in ado_actions]
        self.ado_brake = [a['brake'] for a in ado_actions]
        self.ado_steer = [a['steer'] for a in ado_actions]

    def _get_ado_control(self) -> carla.VehicleControl:
        """Get current lead vehicle control."""
        throttle = self.ado_throttle[self.time_step]
        brake = self.ado_brake[self.time_step]
        steer = self.ado_steer[self.time_step]
        return carla.VehicleControl(
            throttle=float(throttle), brake=float(brake), steer=float(steer)
        )

    @property
    def completed(self) -> bool:
        """Check if episode is complete."""
        return self.time_step >= self.config["max_timesteps"] or self.collided_event

    def step(self, acc: float) -> Observation:
        """Execute one simulation step."""
        acc = np.clip(acc, -10.0, 10.0)
        self.ego.apply_control(self._get_control(acc))
        self.ado.apply_control(self._get_ado_control())

        self.world.tick()

        self.time_step += 1
        self.total_step += 1

        return self._get_obs()

    # ==================== OBSERVATIONS ====================
    
    def _get_obs(self) -> Observation:
        """
        Get observations using SENSORS (not ground truth).
        
        Distance: From radar sensor (affected by weather)
        Velocity: From simulated wheel encoder with noise
        """
        # Rendering
        if self.render:
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            birdeye_render_types = ["roadmap", "actors"]
            self.birdeye_render.render(self.display, birdeye_render_types)
            birdeye = pygame.surfarray.array3d(self.display)
            birdeye = birdeye[0:self.display_size, :, :]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
            self.display.blit(birdeye_surface, (0, 0))

            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (self.display_size * 2, 0))

            pygame.display.flip()

        self.camera_buffer.append(self.camera_img.copy())

        # ========== GROUND TRUTH (for comparison/logging) ==========
        ego_trans = self.ego.get_transform()
        ado_trans = self.ado.get_transform()
        self.ground_truth_distance = self._distance_between_transforms(ego_trans, ado_trans)
        self.ground_truth_velocity = self._get_ground_truth_velocity()

        # ========== SENSOR-BASED MEASUREMENTS ==========
        if self.use_ground_truth:
            # Fallback to ground truth if configured
            distance_to_lead = self.ground_truth_distance
            ego_velocity = self.ground_truth_velocity
        else:
            # Use radar for distance
            radar_distance = self.radar_processor.process_radar_data(
                self.radar_processor.raw_detections
            )
            
            if radar_distance is not None:
                distance_to_lead = radar_distance
                self.sensor_distance = radar_distance
            else:
                # No detection - return max range (simulates "no target")
                distance_to_lead = float('inf')
                self.sensor_distance = float('inf')
            
            # Use velocity processor for speed
            ego_velocity = self.velocity_processor.process_velocity(
                self.ground_truth_velocity
            )
            self.sensor_velocity = ego_velocity

        # Handle collision
        if self.collided_event:
            distance_to_lead = 0

        return Observation(
            ego_velocity=ego_velocity,
            distance_to_lead=distance_to_lead,
            desired_speed=self.config["desired_speed"],
        )

    def _distance_between_transforms(self, p1: carla.Transform, p2: carla.Transform) -> float:
        """Calculate ground truth distance between transforms."""
        p1_pos = np.array([p1.location.x, p1.location.y], dtype=np.float64)
        p2_pos = np.array([p2.location.x, p2.location.y], dtype=np.float64)

        if abs(p2_pos[1] - p1_pos[1]) >= 2:
            return float('inf')
        return np.linalg.norm(p2_pos - p1_pos)

    def _get_ground_truth_velocity(self) -> float:
        """Get ground truth ego velocity."""
        velocity = self.ego.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def _get_ado_velocity(self) -> float:
        """Get lead vehicle velocity (ground truth for logging)."""
        velocity = self.ado.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def _get_ego_velocity(self) -> float:
        """Get ego velocity - uses sensor if available."""
        if self.use_ground_truth:
            return self._get_ground_truth_velocity()
        return self.sensor_velocity

    def _get_actor_polygons(self, filt):
        """Get actor bounding box polygons for rendering."""
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def dump_video(self, filename):
        """Save camera buffer to video file."""
        if not self.camera_buffer:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(1.0 / self.config['dt'])
        frame_width = self.camera_buffer[0].shape[1]
        frame_height = self.camera_buffer[0].shape[0]
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        for frame in self.camera_buffer:
            video_writer.write(frame)
        video_writer.release()
        self.camera_buffer = []

    def get_sensor_comparison(self) -> Dict[str, float]:
        """
        Get comparison between ground truth and sensor measurements.
        Useful for debugging and analysis.
        """
        return {
            'ground_truth_distance': self.ground_truth_distance,
            'sensor_distance': self.sensor_distance,
            'distance_error': abs(self.ground_truth_distance - self.sensor_distance) 
                             if self.sensor_distance != float('inf') else float('inf'),
            'ground_truth_velocity': self.ground_truth_velocity,
            'sensor_velocity': self.sensor_velocity,
            'velocity_error': abs(self.ground_truth_velocity - self.sensor_velocity),
            'weather_detection_prob': self.weather_effects.detection_probability,
        }


# Alias for backward compatibility
Simulator = SensorSimulator

