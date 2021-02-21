import rospy

from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    MAX_BRAKE = 400


    def __init__(
        self,
        vehicle_mass,
        fuel_capacity,
        brake_deadband,
        decel_limit,
        accel_limit,
        wheel_radius,
        wheel_base,
        steer_ratio,
        max_lat_accel,
        max_steer_angle,
    ):
        # Define parameters for PID
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0.0
        max_throttle = 0.2

        # Define parameters for low pass filter
        tau = 0.5
        sample_time = 0.02

        # Set controllers
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed=0.1,
                max_lat_accel=max_lat_accel,
                max_steer_angle=max_steer_angle)
        self.throttle_controller = PID(kp, ji, kd, min_throttle, max_throttle)
        self.vel_lpf = LowPassFilter(tau, sample_time)
    
        # Assign input args to attributes
        self.vehicle_mass = vehcile_mass
        self.decel_limit = decel_limit
        self.wheel_radius = wheel_radius

        # Set timestamp
        self.last_time = rospy.get_time()

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        current_velocity = self.vel_lpf.filt(current_velocity)
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity,
                current_velocity)

        # Compute velocity error
        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity

        # Compute sample time
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(velocity_error, sample_time) 
        brake = 0

        if linear_vel == 0.0 and current_velocity < 0.1:
            throttle = 0
            brake = Controller.MAX_BRAKE
        elif:
            throttle = 0
            decel = max(velocity_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        return throttle, brake, steering
