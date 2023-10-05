#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cameramodels
import numpy as np
import pybullet
import pybullet_data
import skrobot


class Renderer:
    'based on https://github.com/dougsm/mvp_grasp/blob/master/mvp_grasping/src/mvp_grasping/renderer.py'

    def __init__(
            self,
            im_width,
            im_height,
            fov,
            near_plane,
            far_plane,
            pos,
            rot):
        self.im_width = im_width
        self.im_height = im_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.camera_model \
            = cameramodels.PinholeCameraModel.from_fov(
                fov, im_height, im_width)
        aspect = self.im_width / self.im_height
        self.pm = pybullet.computeProjectionMatrixFOV(
            fov, aspect, near_plane, far_plane)

        self.camera_coords = skrobot.coordinates.Coordinates(
            pos=pos,
            rot=rot)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.draw_camera_pos()

    def draw_camera_pos(self):
        pybullet.removeAllUserDebugItems()
        start = self.camera_coords.worldpos()
        end_x = start + self.camera_coords.rotate_vector([0.1, 0, 0])
        pybullet.addUserDebugLine(start, end_x, [1, 0, 0], 3)
        end_y = start + self.camera_coords.rotate_vector([0, 0.1, 0])
        pybullet.addUserDebugLine(start, end_y, [0, 1, 0], 3)
        end_z = start + self.camera_coords.rotate_vector([0, 0, 0.1])
        pybullet.addUserDebugLine(start, end_z, [0, 0, 1], 3)

    def render(self):
        target = self.camera_coords.worldpos() + \
            self.camera_coords.rotate_vector([0, 0, 1.])
        up = self.camera_coords.rotate_vector([0, -1, 0])

        vm = pybullet.computeViewMatrix(
            self.camera_coords.worldpos(), target, up)

        i_arr = pybullet.getCameraImage(
            self.im_width, self.im_height, vm, self.pm,
            shadow=0,
            renderer=pybullet.ER_TINY_RENDERER)

        return i_arr

    def get_depth(self):
        return self.render()[3]

    def get_depth_metres(self, noise=0.001):
        d = self.render()[3]
        # Linearise to metres
        return 2 * self.far_plane * self.near_plane / (
            self.far_plane + self.near_plane - (
                self.far_plane - self.near_plane) * (
                    2 * d - 1)) + np.random.randn(
                        self.im_height, self.im_width) * noise

    def get_rgb(self):
        return self.render()[2]

    def get_seg(self):
        return self.render()[4]
