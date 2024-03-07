#!/usr/bin/env rye-shebang
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
import os
import sys

import cv2
import numpy as np
import PySide2
import rospkg
import rospy
import sensor_msgs.point_cloud2 as pc2
from contact_graspnet_ros.srv import EstimateGrasp
from contact_graspnet_ros.srv import EstimateGraspRequest
from contact_graspnet_ros.srv import EstimateGraspResponse
from contact_graspnet_ros.srv import Inference
from contact_graspnet_ros.srv import InferenceRequest
from contact_graspnet_ros.srv import InferenceResponse
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
from tf.transformations import quaternion_from_euler
from tf.transformations import quaternion_from_matrix
from tf.transformations import quaternion_multiply

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

import config_utils
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps


class GraspInference(object):
    def __init__(self) -> None:
        pkg_path = rospkg.RosPack().get_path('contact_graspnet_ros')
        self.checkpoint_dir = pkg_path + '/checkpoints/scene_test_2048_bs3_hor_sigma_001'
        self.filter_grasps = True

        self.bridge = CvBridge()

        global_config = config_utils.load_config(self.checkpoint_dir, batch_size=1)

        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, self.checkpoint_dir, mode='test')

        camera_info = rospy.wait_for_message('~camera_info', CameraInfo)
        self.k = np.array(camera_info.K).reshape(3, 3)
        # subs = []
        # subs.append(message_filters.Subscriber('~depth', Image))
        # subs.append(message_filters.Subscriber('~rgb', Image))
        # subs.append(message_filters.Subscriber('~segmap', Image))
        rospy.Service('~inference', Inference, self.inference)
        rospy.Service('~estimate_grasps', EstimateGrasp, self.estimate_grasps)

        # sync = message_filters.ApproximateTimeSynchronizer(subs, 1000, 1)
        # sync.registerCallback(self.inference)
        self.pc_colors = None
        self.inferenced = False
        rospy.loginfo('Grasp Inference Ready')

    def estimate_grasps(self, req: EstimateGraspRequest) -> EstimateGraspResponse:
        pc_full = np.array(list(pc2.read_points(req.full_cloud, skip_nans=True, field_names=('x', 'y', 'z'))))
        pc_segments = {
            i: np.array(list(pc2.read_points(segment, skip_nans=True, field_names=('x', 'y', 'z'))))
            for i, segment in enumerate(req.object_clouds)
        }
        scale = 0.9
        pc_full *= scale
        for seg in pc_segments.values():
            seg *= scale
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess,
                                                                                            pc_full,
                                                                                            pc_segments=pc_segments,
                                                                                            filter_grasps=True)

        # self.color = color
        # self.segmap = segmap
        self.pc_full = pc_full
        self.pred_grasps_cam = pred_grasps_cam
        self.scores = scores
        # self.pc_colors = pc_colors
        self.inferenced = True
        resp = EstimateGraspResponse()
        for mat, score in zip(pred_grasps_cam.values(), scores.values()):
            if mat.shape[0] == 0:
                continue
            poses = PoseArray()
            poses.header = req.full_cloud.header
            for i in range(mat.shape[0]):
                pose = Pose()
                offset = np.ravel(mat[i] * np.matrix([0, 0, 0.1, 0]).T)
                pose.position.x = mat.item(i, 0, 3) / scale + offset[0]
                pose.position.y = mat.item(i, 1, 3) / scale + offset[1]
                pose.position.z = mat.item(i, 2, 3) / scale + offset[2]
                pose.orientation = Quaternion(
                    *quaternion_multiply(quaternion_from_matrix(mat[i]), quaternion_from_euler(0, 0, np.pi / 2)))
                poses.poses.append(pose)
            resp.poses.append(poses)
            scores = Float32MultiArray()
            scores.data = list(score)
            resp.scores.append(scores)
        return resp

    # def inference(self, depth_msg, color_msg, segmentation_ids) -> None:
    def inference(self, req: InferenceRequest) -> InferenceResponse:

        try:
            segmap = self.bridge.imgmsg_to_cv2(req.segmap, 'passthrough')
            depth = self.bridge.imgmsg_to_cv2(req.depth, 'passthrough')
            color = self.bridge.imgmsg_to_cv2(req.rgb, desired_encoding='bgr8')
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            rospy.logwarn(e)
            return InferenceResponse()
        scale = 0.9
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth,
                                                                                    self.k,
                                                                                    segmap=segmap,
                                                                                    rgb=color)
        pc_full *= scale
        for seg in pc_segments.values():
            seg *= scale
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess,
                                                                                            pc_full,
                                                                                            pc_segments=pc_segments,
                                                                                            filter_grasps=True)

        self.color = color
        self.segmap = segmap
        self.pc_full = pc_full
        self.pred_grasps_cam = pred_grasps_cam
        self.scores = scores
        self.pc_colors = pc_colors
        self.inferenced = True
        resp = InferenceResponse()
        for mat, score in zip(pred_grasps_cam.values(), scores.values()):
            if mat.shape[0] == 0:
                continue
            poses = PoseArray()
            poses.header = req.depth.header
            for i in range(mat.shape[0]):
                pose = Pose()
                offset = np.ravel(mat[i] * np.matrix([0, 0, 0.1, 0]).T)
                pose.position.x = mat.item(i, 0, 3) / scale + offset[0]
                pose.position.y = mat.item(i, 1, 3) / scale + offset[1]
                pose.position.z = mat.item(i, 2, 3) / scale + offset[2]
                pose.orientation = Quaternion(
                    *quaternion_multiply(quaternion_from_matrix(mat[i]), quaternion_from_euler(0, 0, np.pi / 2)))
                poses.poses.append(pose)
            resp.poses.append(poses)
            scores = Float32MultiArray()
            scores.data = list(score)
            resp.scores.append(scores)
        return resp

    def run(self) -> None:
        while not rospy.is_shutdown():
            if self.inferenced:
                # show_image(self.color, self.segmap)
                visualize_grasps(self.pc_full,
                                 self.pred_grasps_cam,
                                 self.scores,
                                 plot_opencv_cam=True,
                                 pc_colors=self.pc_colors)
            rospy.sleep(1)


if __name__ == '__main__':
    rospy.init_node('grasp_inference')
    app = GraspInference()
    # app.run()
    rospy.spin()
