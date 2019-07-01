import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, action_repeat=3, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, reward_gains=np.array([0, 0, 0, 0, 0, 0])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        
        #self.action_repeat = 3
        self.action_repeat = action_repeat #if action_repeat is None else 1 

        # SRS: Adjust sise according to all features in state
        self.state_size = int(self.action_repeat * (6+3+0*3))
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0.,0.,0., 0.,0.,0.]) 
        
        # Reward gains
        self.reward_gains = reward_gains
        
        
    def get_reward(self):
        
        # Original:
        #"""Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum()
        
        """
        SRS: Modified reward
        Uses whatever target_pos that is sent to tha Task.
        """
        
        #print(self.target_pos)
        
        target_position_3D = self.target_pos[:3]
        #print('tarPos', target_position_3D)
        
        target_position_2D = self.target_pos[:2]
        #print('tarPos', target_position_2D)
        
        target_Euler = self.target_pos[3:6]
        #print('tarVel', target_Euler)
        
        target_velocity = self.target_pos[-3:]
        #print('tarRot', target_velocity)
        
        #print(self.sim.pose[:3])
        
        diff_position_3D = np.sqrt((np.dot(self.sim.pose[:3] - target_position_3D, self.sim.pose[:3] - target_position_3D)).sum())
        diff_position_2D = np.sqrt((np.dot(self.sim.pose[:2] - target_position_2D, self.sim.pose[:2] - target_position_2D)).sum())
        diff_velocity = abs(self.sim.v - target_velocity).sum()
        diff_Euler = np.sqrt((np.dot(self.sim.pose[-3:] - target_Euler, self.sim.pose[-3:] - target_Euler)).sum())
        diff_boundary=min(300-max(self.sim.pose[:3]), min(self.sim.pose[:3])-300)
        
        # reset all rewards
        reward_proximity = reward_position = reward_velocity_close = reward_boundary = reward_e_proximity = reward_rotation_close = 0
        
        Normalization = 0.002
        
        proximity_gain = self.reward_gains[0]
        #reward_proximity = np.power(diff_position_3D + 0.1, -0.7)
        reward_proximity = 1-Normalization*diff_position_3D
        reward_proximity = reward_proximity * proximity_gain 

        proximity_e_gain = self.reward_gains[1] 
        #reward_e_proximity = np.power(diff_position_3D + 0.0001, -0.7)      
        # Make revard_position decrease close the target
        variance = 12
        reward_e_proximity = 1-1/(variance*np.sqrt(2*np.pi)) * np.exp(-1*(diff_position_3D+6)**2 / (2*variance**2) )
        reward_e_proximity = reward_e_proximity * proximity_e_gain
        #print('diff_position_3D: ', diff_position_3D, 'reward_e_proximity: ', reward_e_proximity)
 
        position2D_gain = self.reward_gains[2]
        reward_2D = np.power(diff_position_2D + 0.01, -0.2)
        reward_2D = reward_2D * position2D_gain

        velocity_close_gain = self.reward_gains[3]
        reward_velocity_close = np.power(diff_velocity + 0.0001, -1.2) # MAke revard_velocity increase exponentially closer to target
        reward_velocity_close = reward_velocity_close * np.power(1.4, -diff_position_3D*2) # Make reward_velocity count only very close to the target
        reward_velocity_close = reward_velocity_close * velocity_close_gain

        Euler_close_gain = self.reward_gains[4] 
        reward_Euler_close = np.power(diff_Euler + 0.01, -1.2) # MAke revard_Euler increase exponentially closer to target
        reward_Euler_close = reward_Euler_close * np.power(1.2, -diff_position_3D*2) # Make reward_Euler count only very close 
        reward_Euler_close = reward_Euler_close * Euler_close_gain
        
        boundary_gain = self.reward_gains[5]
        reward_boundary=0
        if diff_boundary<50: reward_boundary=-1
        reward_boundary = reward_boundary * boundary_gain


        # Sum all rewards
        reward = reward_proximity + reward_2D + reward_velocity_close + reward_boundary + reward_e_proximity + reward_Euler_close
        
        #if self.sim.pose[2]<2: reward = -1
            
        return reward
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose[:3])
            """
            Expand the size of the state vector by including the velocity information. 
            You can use any combination of the pose, velocity, and angular velocity
            """
            #SRS: Appending velocity and angular_velocity
            pose_all.append(self.sim.v)
            pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        
        #SRS: Adjust for all features to scored on. Rejects Euler angles
        state = np.concatenate([*[self.sim.pose], *[self.sim.v]] * self.action_repeat)
        #state = np.concatenate([*[self.sim.pose[:3]], *[self.sim.v], *[self.sim.angular_v]] * self.action_repeat)
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state