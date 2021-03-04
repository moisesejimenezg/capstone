#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
DEBUG = False
LOG_LEVEL = rospy.INFO

LABEL_MODE = 0
CLSFY_MODE = 1


class TLDetector(object):
    def __init__(self):
        rospy.init_node("tl_detector", log_level=LOG_LEVEL)

        self.__current_pose = None
        self.__waypoints = None
        self.__waypoints_2d = None
        self.__waypoint_tree = None
        self.__camera_image = None
        self.__lights = []
        self.__has_image = False
        self.__bridge = CvBridge()
        self.__mode = LABEL_MODE
        if self.__mode == LABEL_MODE:
            self.__light_classifier = TLClassifier("wb")
        else:
            self.__light_classifier = TLClassifier()
        self.__listener = tf.TransformListener()
        self.__state = TrafficLight.UNKNOWN
        self.__last_state = TrafficLight.UNKNOWN
        self.__last_wp = -1
        self.__state_count = 0
        self.__classification_done = False

        sub1 = rospy.Subscriber("/current_pose", PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber("/base_waypoints", Lane, self.waypoints_cb)

        """
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        """
        sub3 = rospy.Subscriber(
            "/vehicle/traffic_lights", TrafficLightArray, self.traffic_cb
        )
        sub6 = rospy.Subscriber("/image_color", Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.__config = yaml.load(config_string)

        self.__upcoming_red_light_pub = rospy.Publisher(
            "/traffic_waypoint", Int32, queue_size=1
        )
        rospy.loginfo("TLDetector: Initialized.")
        rospy.spin()

    def pose_cb(self, msg):
        self.__current_pose = msg

    def waypoints_cb(self, waypoints):
        self.__waypoints = waypoints
        if not self.__waypoints_2d:
            self.__waypoints_2d = [
                [waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                for waypoint in waypoints.waypoints
            ]
            self.__waypoint_tree = KDTree(self.__waypoints_2d)

    def __publish_traffic_light_state(self, light_wp, state):
        rospy.logdebug("TLDetector.__publish_traffic_light_state")
        if self.__state != state:
            rospy.logdebug("TLDetector: State changed.")
            self.__state_count = 0
            self.__state = state
        elif self.__state_count >= STATE_COUNT_THRESHOLD:
            rospy.logdebug("TLDetector: State is stable.")
            self.__last_state = self.__state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.__last_wp = light_wp
            self.__upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            rospy.logdebug("TLDetector: Publishing old state.")
            self.__upcoming_red_light_pub.publish(Int32(self.__last_wp))
        self.__state_count += 1

    def traffic_cb(self, msg):
        self.__lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        rospy.logdebug("TLDetector.image_cb")
        self.__has_image = True
        self.__camera_image = msg

        cv_image = self.__bridge.imgmsg_to_cv2(msg, "bgr8")
        light_wp, state = self.__process_traffic_lights()
        if self.__mode == LABEL_MODE and not self.__classification_done:
            self.__classification_done = self.__light_classifier.save_image(
                cv_image, state
            )
            if self.__classification_done:
                rospy.loginfo("TLDetector.image_cb: Done generating labels.")

        """
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        """
        self.__publish_traffic_light_state(light_wp, state)

    def __get_closest_waypoint_index(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.__waypoint_tree.query([x, y], 1)[1]

    def __get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.logdebug("TLDetector.__get_light_state")
        if not self.__has_image:
            self.__prev_light_loc = None
            return False

        cv_image = self.__bridge.imgmsg_to_cv2(self.__camera_image, "bgr8")

        rospy.logdebug("TLDetector: classifying light")
        return self.__light_classifier.get_classification(cv_image)

    def __get_closest_light(self):
        rospy.logdebug("TLDetector.__get_closest_light")
        closest_light = None
        line_index = 0
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.__config["stop_line_positions"]
        if self.__current_pose and self.__waypoint_tree and self.__waypoints:
            rospy.logdebug("TLDetector: Looking for closest light")
            car_x = self.__current_pose.pose.position.x
            car_y = self.__current_pose.pose.position.y
            car_index = self.__get_closest_waypoint_index(car_x, car_y)

            distance = len(self.__waypoints.waypoints)
            for i, light in enumerate(self.__lights):
                stop_line = stop_line_positions[i]
                stop_line_index = self.__get_closest_waypoint_index(
                    stop_line[0], stop_line[1]
                )
                current_distance = stop_line_index - car_index
                if current_distance >= 0 and current_distance < distance:
                    distance = current_distance
                    closest_light = light
                    line_index = stop_line_index
        rospy.logdebug("TLDetector: Closest light at index: " + str(line_index))
        return closest_light, line_index

    def __process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.logdebug("TLDetector.__process_traffic_lights")

        closest_light, line_index = self.__get_closest_light()

        if closest_light:
            if self.__mode == CLSFY_MODE:
                state = self.__get_light_state(closest_light)
            elif self.__mode == LABEL_MODE:
                state = closest_light.state
            rospy.logdebug(
                "TLDetector: Publishing index: "
                + str(line_index)
                + " and state: "
                + str(state)
            )
            return line_index, state
        rospy.logdebug("TLDetector: Publishing default")
        return -1, TrafficLight.UNKNOWN


if __name__ == "__main__":
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr("TLDetector: Could not start traffic node.")
