import os

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

        self.bridge_half_width = bridge_width * 0.5

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
        bridge_center_local = torch.tensor([1.5, 0.0, bridge_height * 0.5 + bridge_clearance], device=self.device)
        robot_spawn_local = torch.tensor([0.2, 0.0, self.base_init_state[2].item()], device=self.device)

        self.bridge_center_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.bridge_top_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        num_per_row = int(self.num_envs ** 0.5)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            bridge_world = self.env_origins[i] + bridge_center_local
            bridge_pose = gymapi.Transform()
            bridge_pose.p = gymapi.Vec3(float(bridge_world[0]), float(bridge_world[1]), float(bridge_world[2]))
            bridge_handle = self.gym.create_actor(env_handle, bridge_asset, bridge_pose, 'bridge', i, 0, 0)
            self.gym.set_rigid_body_color(
                env_handle,
                bridge_handle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.35, 0.35, 0.40),
            )

            spawn_world = self.env_origins[i] + robot_spawn_local
            start_pose.p = gymapi.Vec3(float(spawn_world[0]), float(spawn_world[1]), float(spawn_world[2]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.bridge_center_y[i] = bridge_world[1]
            self.bridge_top_z[i] = bridge_world[2] + bridge_height * 0.5

            self.envs.append(env_handle)
            self.bridge_handles.append(bridge_handle)
            self.actor_handles.append(actor_handle)

            if i == 0:
                print('bridge actor created')
                print('bridge pose', [float(bridge_world[0]), float(bridge_world[1]), float(bridge_world[2])])
                print('robot spawn pose', [float(spawn_world[0]), float(spawn_world[1]), float(spawn_world[2])])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        if not hasattr(self.cfg, 'debug') or not self.cfg.debug.enable:
            return

        print_interval_steps = max(1, int(self.cfg.debug.print_interval_s / self.dt))
        if self.common_step_counter % print_interval_steps != 0:
            return

        env_id = 0
        base_pos = self.base_pos[env_id]
        lateral_offset = base_pos[1] - self.bridge_center_y[env_id]
        dropped_off_bridge = (base_pos[2] < self.bridge_top_z[env_id] - self.cfg.debug.fall_height_threshold) or (
            torch.abs(lateral_offset) > (self.bridge_half_width + self.cfg.debug.bridge_side_margin)
        )

        print(
            '[go2_bridge debug] base position:',
            [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])],
            '| lateral offset:',
            float(lateral_offset),
            '| dropped:',
            bool(dropped_off_bridge),
        )
