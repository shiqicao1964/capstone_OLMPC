#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped ,Pose, Twist
import numpy as np
from std_msgs.msg import Bool
import math
flag = Bool()
def flag_cb(msg):
    global flag
    flag = msg


if __name__ == "__main__":
    rospy.init_node("reference_publisher")
    rate = rospy.Rate(1)
    reference_pub = rospy.Publisher("reference", Twist, queue_size=1)
    reference_pub_2 = rospy.Publisher("reference2", Pose, queue_size=1)
    rospy.Subscriber("start_time", Bool, callback = flag_cb)
    
    msg = Twist()
    msg_2 = Pose()




    discretization_dt = 0.1
    radius = 8
    z = 3
    lin_acc = 4
    clockwise = True
    yawing = True
    v_max = 8
    v_average = 5
    sim_t = 30

    t_speedup = v_average/lin_acc
    t_speeddown = t_speedup
    t_uniform_circular = sim_t - t_speedup*2
    angle_acc = lin_acc / radius  # rad/s^2
    t_speedup_points = np.linspace(start = 0, stop = t_speedup, num = int(t_speedup/discretization_dt)+1)
    angle_points_1 = 0.5 * angle_acc * t_speedup_points**2
    anglevel_points_1 = angle_acc * t_speedup_points
    t_uniform_circular_points = np.linspace(start= discretization_dt, stop=t_uniform_circular, num=int(t_uniform_circular/discretization_dt))
    angle_points_2 = angle_points_1[-1] + t_uniform_circular_points * v_average/radius
    anglevel_points_2 = t_uniform_circular_points * 0 + anglevel_points_1[-1]
    t_speeddown_points = np.linspace(start = discretization_dt, stop = t_speeddown, num = int(t_speeddown/discretization_dt))
    angle_points_3 = angle_points_2[-1] + v_average/radius * t_speeddown_points - 0.5 * angle_acc * t_speeddown_points**2
    anglevel_points_3 = anglevel_points_2[-1] - angle_acc * t_speeddown_points

    angle_points = np.concatenate((angle_points_1,angle_points_2,angle_points_3))
    anglevel_points = np.concatenate((anglevel_points_1,anglevel_points_2,anglevel_points_3))
    pos_traj_x = radius * np.sin(angle_points)
    pos_traj_y = radius * np.cos(angle_points)
    pos_traj_z = np.ones_like(pos_traj_x) * z
    vel_traj_x = anglevel_points * radius * np.cos(angle_points)
    vel_traj_y = anglevel_points * radius * np.sin(angle_points)


    index = 0
    while(1):
        if (flag):
            msg_2.position.x = 0
            msg_2.position.y = 2.2
            msg_2.position.z = 3.2
            msg_2.orientation.x = 0
            msg_2.orientation.y = 0
            msg_2.orientation.z = 0
            msg_2.orientation.w = 0
            msg.linear.x = 1
            msg.linear.y = 2
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = 0
            reference_pub.publish(msg)
            reference_pub_2.publish(msg_2)
            index = index + 1
        rate.sleep()
