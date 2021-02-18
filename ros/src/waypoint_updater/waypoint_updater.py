#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.__current_pose = None
        self.__base_waypoints = None
        self.__waypoints_2d = None
        self.__waypoint_tree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.__final_waypoints_publisher = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.__step()

    def __step(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.__current_pose and self.__base_waypoints and self.__waypoint_tree:
                next_waypoint_index = self.__get_next_waypoint_index()
                self.__publish_waypoints(next_waypoint_index)
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

        dot_product = np.dot(closest_point_vector - previous_point_vector, pose_vector - closest_point_vector)

        if dot_product > 0:
            closest_index = (closest_index + 1) % len(self.__waypoints_2d)
        return closest_index

    def __publish_waypoints(self, next_waypoint_index):
        lane = Lane()
        lane.header = self.__base_waypoints.header
        lane.waypoints = self.__base_waypoints.waypoints[next_waypoint_index: next_waypoint_index + LOOKAHEAD_WPS]
        self.__final_waypoints_publisher.publish(lane)
        pass

    def pose_cb(self, msg):
        self.__current_pose = msg
        pass

    def waypoints_cb(self, waypoints):
        self.__base_waypoints = waypoints
        if not self.__waypoints_2d:
            self.__waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.__waypoint_tree = KDTree(self.__waypoints_2d)
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
