#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

"""
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
"""

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node("waypoint_updater", log_level=rospy.WARN)

        self.__current_pose = None
        self.__base_waypoints = None
        self.__waypoints_2d = None
        self.__waypoint_tree = None
        self.__light_index = -1

        rospy.Subscriber("/current_pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/base_waypoints", Lane, self.waypoints_cb)
        rospy.Subscriber("/traffic_waypoint", Int32, self.traffic_cb)

        self.__final_waypoints_publisher = rospy.Publisher(
            "/final_waypoints", Lane, queue_size=1
        )

        rospy.loginfo("WaypointUpdater: Initialized.")
        self.__step()

    def __step(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.__current_pose and self.__base_waypoints and self.__waypoint_tree:
                self.__publish_waypoints()
            rate.sleep()

    def __get_next_waypoint_index(self):
        x = self.__current_pose.pose.position.x
        y = self.__current_pose.pose.position.y
        closest_index = self.__waypoint_tree.query([x, y], 1)[1]

        closest_point = self.__waypoints_2d[closest_index]
        previous_point = self.__waypoints_2d[closest_index - 1]

        closest_point_vector = np.array(closest_point)
        previous_point_vector = np.array(previous_point)
        pose_vector = np.array([x, y])

        dot_product = np.dot(
            closest_point_vector - previous_point_vector,
            pose_vector - closest_point_vector,
        )

        if dot_product > 0:
            closest_index = (closest_index + 1) % len(self.__waypoints_2d)
        return closest_index

    def __extract_reference_waypoints(self, next_waypoint_index):
        return self.__base_waypoints.waypoints[
            next_waypoint_index : next_waypoint_index + LOOKAHEAD_WPS
        ]

    def __generate_decelerating_waypoints(self, next_waypoint_index):
        rospy.loginfo("WaypointUpdater: Decelerating waypoints.")
        rospy.logdebug(
            "WaypointUpdater: Next Waypoint Index is: " + str(next_waypoint_index)
        )
        reference_waypoints = self.__extract_reference_waypoints(next_waypoint_index)
        new_waypoints = []
        for i, wp in enumerate(reference_waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_index = max(self.__light_index - next_waypoint_index - 2, 0)
            distance = self.distance(reference_waypoints, i, stop_index)
            velocity = math.sqrt(2 * MAX_DECEL * distance)
            if velocity < 1.0:
                velocity = 0
            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            new_waypoints.append(p)
        return new_waypoints

    def __next_stop_line_is_within_range(self, next_waypoint_index):
        return (
            self.__light_index == -1
            or self.__light_index >= next_waypoint_index + LOOKAHEAD_WPS
        )

    def __generate_lane(self, next_waypoint_index):
        lane = Lane()
        lane.header = self.__base_waypoints.header
        if self.__next_stop_line_is_within_range(next_waypoint_index):
            rospy.loginfo("WaypointUpdater: Using reference waypoints.")
            lane.waypoints = self.__extract_reference_waypoints(next_waypoint_index)
        else:
            lane.waypoints = self.__generate_decelerating_waypoints(next_waypoint_index)
        return lane

    def __publish_waypoints(self):
        rospy.logdebug("WaypointUpdater: Publish waypoints.")
        next_waypoint_index = self.__get_next_waypoint_index()
        lane = self.__generate_lane(next_waypoint_index)
        self.__final_waypoints_publisher.publish(lane)

    def pose_cb(self, msg):
        self.__current_pose = msg

    def waypoints_cb(self, waypoints):
        self.__base_waypoints = waypoints
        if not self.__waypoints_2d:
            self.__waypoints_2d = [
                [waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                for waypoint in waypoints.waypoints
            ]
            self.__waypoint_tree = KDTree(self.__waypoints_2d)

    def traffic_cb(self, msg):
        self.__light_index = msg.data
        rospy.logdebug(
            "WaypointUpdater: Next traffic light at " + str(self.__light_index)
        )

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        )
        for i in range(wp1, wp2 + 1):
            dist += dl(
                waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position
            )
            wp1 = i
        return dist


if __name__ == "__main__":
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr("Could not start waypoint updater node.")
