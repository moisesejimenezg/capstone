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
        self.__yaw_controller = YawController(
            wheel_base,
            steer_ratio,
            min_speed=0.1,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle,
        )
        self.__throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        self.__velocity_lowpassfilter = LowPassFilter(tau, sample_time)

        # Assign input args to attributes
        self.__vehicle_mass = vehicle_mass
        self.__decel_limit = decel_limit
        self.__wheel_radius = wheel_radius

        # Set timestamp
        self.__last_time = rospy.get_time()

    @property
    def yaw_controller(self):
        assert self.__yaw_controller is not None
        return self.__yaw_controller

    @property
    def throttle_controller(self):
        assert self.__throttle_controller is not None
        return self.__throttle_controller

    @property
    def velocity_lowpassfilter(self):
        assert self.__velocity_lowpassfilter is not None
        return self.__velocity_lowpassfilter

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        current_velocity = self.velocity_lowpassfilter.filt(current_velocity)
        steering = self.yaw_controller.get_steering(
            linear_velocity, angular_velocity, current_velocity
        )

        # Compute velocity error
        velocity_error = linear_velocity - current_velocity
        # self.last_velocity = current_velocity

        # Compute sample time
        current_time = rospy.get_time()
        sample_time = current_time - self.__last_time
        self.__last_time = current_time

        throttle = self.throttle_controller.step(velocity_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_velocity < 0.1:
            throttle = 0
            brake = Controller.MAX_BRAKE
        elif throttle < 0.1 and velocity_error < 0.0:
            throttle = 0
            decel = max(velocity_error, self.__decel_limit)
            brake = abs(decel) * self.__vehicle_mass * self.__wheel_radius

        return throttle, brake, steering
