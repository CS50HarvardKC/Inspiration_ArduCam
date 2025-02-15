"""
To aggregate/integrate the various lower-level files (sensor fusion, GPS, IMU),
and create a single list of values to be sent to the Arduino to actuate the motors.
"""

import time
import threading
import queue
import math
from GNC.Nav_Core import gis_funcs
from GNC.Control_Core import sensor_fuse, GPS

from API.Motors import t200

class MotorCore():
    def __init__(self, port = "/dev/ttyACM0"):
        self.t200 = t200.T200(port="/dev/ttyACM0")
        # TODO: Put correct heading offset
        self.sensor_fuse = sensor_fuse.SensorFuse(enable_filter=False, heading_offset=0)
        self.position_data = {'current_position' : None, 'current_heading' : None, 'current_velocity' : None}

        self.desired_position = [None, None] # lat, lon
        self.desired_heading = None # in degrees
    
    """
    ----------------- RUDIMENTARY FUNCTIONS [PROVEN BASED ON TESTS] -----------------
    """

    """
    As of 2/10/2025, Barco Polo's motor configuration is in the following:
    - Stern port thruster at positive PWM levels will make the boat move clockwise.
    - Stern starboard thruster at positive PWM levels will make the boat move counterclockwise.
    - Aft port thruster at positive PWM levels will make the boat move counterclockwise.
    - Aft starboard at positive PWM levels will make the boat move counterclockwise.

    When PWMs are greater than 1500 for Blue Robotics thrusters, the thrusters spin clockwise.
    Thrusters with a clockwise prop have a forward thrust vector when spinning clockwise.
    Thrusters with a counter clockwise prop have a backward thrust vector when spinning clockwise.
    """
    
    def surge(self, magnitude):
        """Configures for forward (positive magnitude) or backward (negative magnitude) movement"""
        self.t200.set_thrusters(-magnitude, -magnitude, magnitude, -magnitude)

    def stay(self):
        """Sets all motors to no power."""
        self.t200.set_thrusters(0,0,0,0)

    def stop(self):
        """Stop motors."""
        self.t200.stop_thrusters()

    def slide(self, magnitude):
        """Sliding (strafing) in horizontal direction without rotating, positive is left, negative is right."""
        self.t200.set_thrusters(magnitude,-magnitude,magnitude,magnitude)

    def rotate(self, magnitude):
        """Yaw/rotation, positive magnitude is clockwise, negative is counterclockwise."""
        self.t200.set_thrusters(-magnitude,magnitude,magnitude,magnitude)

    """
    --------------------------------------------------------
    """

    """
    ----------------- FUNCTIONS WITH GPS WAYPOINT NAVIGATION/Kalman Filter/Control Loop [NEEDS TESTING] -----------------
    """

    def polar_waypoint_navigation(self, distance_theta, heading):
        """
        Navigate to a given point that is a certain number of meters away along a certain heading.
        Will rotate and then move the set distance.

        Args:
            distance_theta (float): Distance to move (in meters)
            heading (float): New absolute heading (degrees)
        """
        desired_lat, desired_lon = gis_funcs.destination_point(
            self.position_data["current_position"][0],
            self.position_data["current_position"][1],
            heading,
            distance_theta
        )

        self.desired_position = (desired_lat, desired_lon)
        self.desired_heading = heading

    def lat_lon_navigation(self, lat, lon):
        """
        Navigate to a new (lat, lon) GPS coordinate. Automatically calculates best heading.

        Args:
            lat (float): Desired GPS latitude coordinate.
            lon (float): Desired GPS longitude coordinate.
        """
        self.desired_position = (lat, lon)
        self.desired_heading = gis_funcs.bearing(
            self.position_data["current_position"][0], self.position_data["current_position"][1], lat, lon
        )

    def cartesian_vector_navigation(self, x, y):
        """
        Move along a certain 2-D vector. Automatically calculates best heading.

        Args:
            x (float): Desired displacement in meters along strafe direction.
            y (float): Desired displacement in meters along surge direction.
        """
        vector_distance = round(math.sqrt(x^2 + y^2), 2)
        vector_theta = round(math.degrees(math.atan2(y/x)), 2)
        self.polar_waypoint_navigation(vector_distance, vector_theta)

    def update_position(self):
        # TODO: Should test this before full file test
        self.position_data = {
            'current_position' : self.sensor_fuse.get_position(),
            'current_heading' : self.sensor_fuse.get_heading(),
            'current_velocity' : self.sensor_fuse.get_velocity()
        }

    def calc_motor_power(self, send_queue, calculate_rate, stop_event):
        while not stop_event.is_set():
            self.update_position()
            motor_values = [0, 0, 0, 0]
            # TODO: Calculation logic here
            send_queue.put(motor_values)
            time.sleep(calculate_rate)

    def control_loop(self, value_queue, send_rate, stop_event):
        """
        Takes the values calculated by calc_motor_power(), and sends them to the T200s.
        """
        while not stop_event.is_set():
            try:
                value = value_queue.get(timeout=send_rate)
                stern_port, stern_starboard, aft_port, aft_starboard = value[1], value[2], value[3], value[4]
                self.t200.set_thrusters(stern_port, stern_starboard,aft_port, aft_starboard)
            except queue.Empty:
                continue

    # Calculate motor power thread will send to the control_loop thread. Will have a constant point where we want to go
    # (self.desired_position, heading, etc.), which will be passed into calc_motor_power(). Will deal with specifics for each later.

    def main(self, calculate_rate=0.1, send_rate=0.1, duration=10):
        send_queue = queue.Queue()
        stop_event = threading.Event()

        control_loop_instance = threading.Thread(target=self.control_loop, args=(send_queue, send_rate, stop_event))
        control_loop_instance.daemon = True # Ensure this thread exits when main program exits.
        control_loop_instance.start()

        calc_motor_power_instance = threading.Thread(target=self.calc_motor_power, args=(send_queue, calculate_rate, stop_event))
        calc_motor_power_instance.daemon = True
        calc_motor_power_instance.start()

        if duration == None:
            # Arbitrary number
            duration = 100

        time.sleep(duration)

        stop_event.set()

        calc_motor_power_instance.join()
        control_loop_instance.join()

    """
    --------------------------------------------------------
    """

    