#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import message_filters
import rospy
from contact_graspnet_ros.srv import Inference
from contact_graspnet_ros.srv import InferenceRequest
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import Image
from uois.srv import Inference as SegInference
from uois.srv import InferenceRequest as SegInferenceRequest


def call_service(depth, rgb):
    rospy.loginfo('Call Service')
    req = SegInferenceRequest()
    req.depth = depth
    req.rgb = rgb
    resp = rospy.ServiceProxy('~segment', SegInference)(req)
    req2 = InferenceRequest()
    req2.depth = depth
    req2.rgb = rgb
    req2.segmap = resp.segmap
    resp2 = rospy.ServiceProxy('~inference', Inference)(req2)
    pub = rospy.Publisher('~grasps', PoseArray, queue_size=1, latch=True)
    for grasps in resp2.poses:
        pub.publish(grasps)
    rospy.loginfo('Done')


if __name__ == '__main__':
    rospy.init_node('grasp_inference')
    subs = []
    subs.append(message_filters.Subscriber('~depth', Image))
    subs.append(message_filters.Subscriber('~rgb', Image))

    sync = message_filters.ApproximateTimeSynchronizer(subs, 1000, 1)
    sync.registerCallback(call_service)
    rospy.spin()
