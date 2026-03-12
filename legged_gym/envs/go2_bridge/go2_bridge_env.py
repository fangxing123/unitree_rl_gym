import os

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2BridgeRobot(LeggedRobot):
    """GO2 bridge task environment with a static narrow bridge actor."""

    def _create_envs(self):
        """Creates robot envs and an explicit static bridge actor per env."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        bridge_length = 3.5
        bridge_width = 0.40
        bridge_height = 0.10
        bridge_options = gymapi.AssetOptions()
        bridge_options.fix_base_link = True
        bridge_options.disable_gravity = True
        bridge_asset = self.gym.create_box(self.sim, bridge_length, bridge_width, bridge_height, bridge_options)

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.bridge_handles = []
        self.envs = []

        bridge_clearance = 0.05
        bridge_center_local = np.array([1.5, 0.0, bridge_height * 0.5 + bridge_clearance], dtype=np.float32)
        robot_spawn_local = np.array([0.20, 0.0, float(self.base_init_state[2].item())], dtype=np.float32)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            bridge_world = self.env_origins[i].cpu().numpy() + bridge_center_local
            bridge_pose = gymapi.Transform()
            bridge_pose.p = gymapi.Vec3(float(bridge_world[0]), float(bridge_world[1]), float(bridge_world[2]))
            bridge_handle = self.gym.create_actor(env_handle, bridge_asset, bridge_pose, "bridge", i, 0, 0)
            self.gym.set_rigid_body_color(
                env_handle,
                bridge_handle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.35, 0.35, 0.40),
            )

            spawn_world = self.env_origins[i].cpu().numpy() + robot_spawn_local
            start_pose.p = gymapi.Vec3(float(spawn_world[0]), float(spawn_world[1]), float(spawn_world[2]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.bridge_handles.append(bridge_handle)
            self.actor_handles.append(actor_handle)

            if i == 0:
                print("bridge actor created")
                print("bridge pose", [float(bridge_world[0]), float(bridge_world[1]), float(bridge_world[2])])
                print("robot spawn pose", [float(spawn_world[0]), float(spawn_world[1]), float(spawn_world[2])])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
