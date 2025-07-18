#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
# from utils import nearest_point
from mpc.utils import nearest_point

# TODO CHECK: include needed ROS msg type headers and libraries
import csv
import os
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from scipy.interpolate import splev, splprep
from visualization_msgs.msg import Marker, MarkerArray
import yaml

@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic # 8

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0]) # default [0.01, 100.0] # [0.01, 10.0] on real car
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0]) # default [0.01, 100.0] # [0.01, 10.0] on real car
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # [x, y, v, yaw] [13.5, 13.5, 5.5, 13.0]
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # [x, y, v, yaw] [13.5, 13.5, 5.5, 13.0]
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.12  # dist step [m] kinematic # default 0.03. Distance between waypoints
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0
    # delta is the steering angle, 
    # v is the velocity, 
    # yaw is the heading angle, 
    # yawrate is the rate of change of the heading angle, 
    # beta is the side slip angle

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')

        config_path = '/home/angli/sim_ws/src/mpc/config/mpc_config.yaml'
        # config_path = '/home/nvidia/f1tenth_ws/src/mpc/config/mpc_config.yaml'
        with open(config_path, 'r') as file:
            config_file = yaml.safe_load(file)

        self.sim = True
        if not self.sim:
            odom_topic = 'pf/viz/inferred_pose'
        else:
            odom_topic = '/ego_racecar/odom'
        
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        self.odom_sub = self.create_subscription(PoseStamped if not self.sim else Odometry, odom_topic, self.pose_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Visualization publisher
        self.viz_waypoints_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.viz_target_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.viz_ref_traj_pub = self.create_publisher(MarkerArray, '/visualization_ref_traj', 10)

        # TODO: get waypoints here
        # folder_path = '/home/nvidia/f1tenth_ws/src/mpc/waypoints'
        folder_path = '/home/angli/sim_ws/src/mpc/waypoints'
        # filename = 'Modified_Points_2024-04-09_19-33.csv' #too many waypoints
        # filename = 'Modified_Points_2024-04-10_14-16.csv' #too many waypoints
        filename = 'Modified_Points_2024-04-11_17-05.csv'
        self.csv_file_path = os.path.join(folder_path, filename)

        # self.csv_file_path = config_file.get('csv_file_path')

        waypoints = []
        with open(self.csv_file_path, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None) # skip the headers
            for row in csv_reader:
                waypoints.append({
                    'posX': float(row[0]),
                    'posY': float(row[1]),
                    'yaw': float(row[2]),
                    'v': float(row[3])
                })
        self.waypoints = np.array([np.array([wp['posX'], wp['posY'], wp['yaw'], wp['v']]) for wp in waypoints])

        # For discontinuity in yaw angle
        # for i in range(1, len(self.waypoints)):
        #     if self.waypoints[i][2] - self.waypoints[i-1][2] > 4.5:
        #         print("> 4.5",i)
        #         self.waypoints[i][2] -= 2*np.pi
        #     if self.waypoints[i][2] - self.waypoints[i-1][2] < -4.5:
        #         print("< 4.5",i)
        #         self.waypoints[i][2] += 2*np.pi
        
        self.visualize_waypoints(self.waypoints)

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        self.prev_state = None
        self.prev_speed_output = None

        # initialize MPC problem
        self.mpc_prob_init()

    def pose_callback(self, pose_msg):
        # TODO: extract pose from ROS msg
        vehicle_state = State()

        if self.sim:
            vehicle_state.x = pose_msg.pose.pose.position.x
            vehicle_state.y = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            vehicle_state.x = pose_msg.pose.position.x
            vehicle_state.y = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        yaw = euler[2]
        # For discontinuity in yaw angle
        if self.prev_state is not None:
            if yaw - self.prev_state.yaw > 4.5:
                yaw -= 2*np.pi
            if yaw - self.prev_state.yaw < -4.5:
                yaw += 2*np.pi

        vehicle_state.yaw = yaw

        # if self.prev_state is not None:
        #     vehicle_state.v = np.sqrt((vehicle_state.x - self.prev_state.x)**2 + (vehicle_state.y - self.prev_state.y)**2)/self.config.DTK

        ### The method above assumes that we get perfect state, but there might be differnece between actual speed and the speed calculated from the waypoints
        # Use last speed output as the current speed
        vehicle_state.v = self.prev_speed_output if self.prev_speed_output is not None else 0.0

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_x = self.waypoints[:, 0]
        ref_y = self.waypoints[:, 1]
        ref_yaw = self.waypoints[:, 2]
        # ref_yaw = np.sin(self.waypoints[:, 2])
        ref_v = self.waypoints[:, 3]

        ### Use calculated spline and velocity
        # ref_x = self.x_spline
        # ref_y = self.y_spline
        # ref_v = self.v

        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        # x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, np.sin(vehicle_state.yaw)]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)
        if self.oa is None or self.odelta_v is None:
            self.prev_state = None
            return
        # TODO: publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer_output
        drive_msg.drive.speed = speed_output
        print(f"Steering Angle: {steer_output}, Speed: {speed_output}, yaw: {vehicle_state.yaw}")
        self.drive_pub.publish(drive_msg)

        self.prev_state = vehicle_state
        self.prev_speed_output = speed_output

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )

        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        error = self.xk - self.ref_traj_k # 4*9
        # error[3, error[3,:] > np.pi] -= 2*np.pi 
        # error[3, error[3,:] < -np.pi] += 2*np.pi

        objective += cvxpy.quad_form(cvxpy.vec(error), Q_block)
        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        #       (4, 9)  (32, 32)   (32, 16)  (2, 8)       (32,)

        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]

        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK

        constraints += [cvxpy.abs(cvxpy.diff(self.uk[1, :]))/self.config.DTK<=self.config.MAX_DSTEER]
        # for t in range(self.config.TK - 1):
        #     constraints += [cvxpy.abs(self.uk[1, t + 1] - self.uk[1, t]) <= self.config.MAX_DSTEER * self.config.DTK]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER

        # Add initial state constraint
        constraints += [self.xk[:, 0]==self.x0k]

        # Add velocity constraints
        constraints += [self.xk[2, :] <= self.config.MAX_SPEED]
        constraints += [self.xk[2, :] >= self.config.MIN_SPEED]
        
        # Add acceleration constraints
        constraints += [self.uk[0, :] <= self.config.MAX_ACCEL]
        constraints += [self.uk[0, :] >= -self.config.MAX_ACCEL]
        
        # Add steering angle constraints
        constraints += [self.uk[1, :] <= self.config.MAX_STEER]
        constraints += [self.uk[1, :] >= self.config.MIN_STEER]

        # constraints += [self.xk[3, :] - self.ref_traj_k[3, :] <= np.pi]
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def nearest_point(self, point, trajectory):
        """
        Return the nearest point along the given piecewise linear trajectory.
        Args:
            point (numpy.ndarray, (2, )): (x, y) of current pose
            trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
                NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
        Returns:
            nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
            nearest_dist (float): distance to the nearest point
            t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
            i (int): index of nearest point in the array of trajectory waypoints
        """
        diffs = trajectory[1:,:] - trajectory[:-1,:]
        l2s   = diffs[:,0]**2 + diffs[:,1]**2
        dots = np.empty((trajectory.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
        projections = trajectory[:-1,:] + (t*diffs.T).T
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp*temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = self.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]
        self.visualize_ref_traj(ref_traj)
        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    
    def visualize_waypoints(self, waypoints):
        """
        waypoints: np.array, first two columns are x, y positions
        """
        marker_array = MarkerArray()
        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.150
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
            marker.color.r = 1.0  
            marker.color.g = 1.0
            marker.color.b = 1.0  
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = waypoint[0]
            marker.pose.position.y = waypoint[1]
            marker.pose.position.z = 0.0
            # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes
            marker_array.markers.append(marker)

        self.viz_waypoints_pub.publish(marker_array)

    def visualize_target_point(self, waypoints, target_point):
        """
        waypoints: np.array, first two columns are x, y positions
        target_point: int, index of the target waypoint
        """
        waypoint = waypoints[target_point]
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.15
        target_marker.scale.y = 0.15
        target_marker.scale.z = 0.5
        target_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        target_marker.color.r = 255.0 
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0 
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = waypoint[0]
        target_marker.pose.position.y = waypoint[1]
        target_marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_target_pub.publish(target_marker)

    def visualize_ref_traj(self, ref_traj):
        marker_array = MarkerArray()
        for i in range(ref_traj.shape[1]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.150
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = ref_traj[0, i]
            marker.pose.position.y = ref_traj[1, i]
            marker.pose.position.z = 0.0
            marker_array.markers.append(marker)
        # print("ref_traj: ", ref_traj)
        self.viz_ref_traj_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()