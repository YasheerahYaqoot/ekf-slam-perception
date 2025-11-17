from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt

def motion_model(u, dt, ekf_state, vehicle_parameters):
    
    #----------Vehicle Parameters
    H = vehicle_parameters['H']
    L = vehicle_parameters['L']
    a = vehicle_parameters['a']
    b = vehicle_parameters['b']
    
    #----------Inputs
    Ve = u[0]
    alpha = u[1]
    Vc = Ve/(1-(np.tan(alpha)*(H/L)))
    
    #----------EKF States
    states = ekf_state['x']
    x = states[0]
    y = states[1]
    phi = states[2]
    
    #----------Motion Model
    mm_x = dt*(Vc*np.cos(phi)-(Vc/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    mm_y = dt*(Vc*np.sin(phi)+(Vc/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    mm_phi = dt*(Vc/L)*np.tan(alpha)
    pose = np.array([
                     [mm_x],
                     [mm_y],
                     [mm_phi]
                    ])
    
    #----------Jacobian Matrix 'G'
    G11 = 1; G12 = 0; G13 = -dt*Vc*(np.sin(phi)+(1/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    G21 = 0; G22 = 1; G23 = dt*Vc*(np.cos(phi)-(1/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    G31 = 0; G32 = 0; G33 = 1
    G = np.array([
                  [G11, G12, G13],
                  [G21, G22, G23],
                  [G31, G32, G33]
                 ])
    
    return pose, G

def odom_predict(u, dt, ekf_state, vehicle_parameters, sigmas):
    
    #----------EKF States
    states = ekf_state['x']
    states = np.reshape(states, (-1, 1))  # Transform to column vector (2D)
    
    #----------EKF Covariance
    covariance = ekf_state['P']
    
    #----------Motion and Jacobian G
    pose, G = motion_model(u, dt, ekf_state, vehicle_parameters)
    
    #----------State Noise 'R'
    R = np.diag([sigmas['xy']**2, sigmas['xy']**2, sigmas['phi']**2])
    
    #----------Adjustment Matrix 'F'
    pose_dimensions = 3
    map_dimensions = states.shape[0]-3
    F =  np.hstack((np.eye(pose_dimensions), np.zeros((3, map_dimensions))))
    
    #----------New States (New Pose)
    new_states = states + (F.T @ pose)
    new_states = np.squeeze(new_states)
    
    #----------Adjusted Jacobian Matrix 'G'
    G_pose = np.hstack((G, np.zeros((3, map_dimensions))))
    G_map = np.hstack((np.zeros((map_dimensions, 3)), np.eye(map_dimensions)))
    G = np.vstack((G_pose, G_map))
    
    #----------Adjusted Noise Matrix 'R'
    R = F.T @ R @ F
    
    #----------New Covariance
    new_covariance = (G @ covariance @ G.T) + R
    new_covariance = slam_utils.make_symmetric(new_covariance)

    #----------New States and Covariance
    ekf_state['x'] = new_states
    ekf_state['P'] = new_covariance

    return ekf_state


def gps_update(gps, ekf_state, sigmas):

    # ---------- EKF Covariance
    P = ekf_state['P']        
    dim  = P.shape[0]-2         
    H = np.concatenate((np.eye(2), np.zeros((2, dim))), axis=1) 
    
    # ---------- Residual
    r = np.transpose(np.subtract(gps, ekf_state['x'][:2]))    
    
    # ---------- Noise Covariance       
    Q = (sigmas['gps']**2)*(np.eye(2))
    
    # ---------- Covariance
    S = H @ P @ H.T + Q       
    S_inv = slam_utils.invert_2x2_matrix(S)
    
    # ---------- Mahalanobis distance of the residual
    d = r.T @ S_inv @ r         
    
    # ---------- Exclude the erroneous measurement from the GPS data         
    if d <= chi2.ppf(0.999, 2):
        K = P @ H.T @ S_inv     
        ekf_state['x'] = ekf_state['x'] + np.squeeze(K @ r)
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])            
        P_temp = (np.eye(P.shape[0]) - K @ H) @ P
        ekf_state['P'] = slam_utils.make_symmetric(P_temp)
         
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    #-----------EKF States
    state_vector = ekf_state['x'].copy()
    state_vec_x = state_vector[0]
    state_vec_y = state_vector[1]
    state_vec_phi = state_vector[2]
    state_vec_phi= slam_utils.clamp_angle(state_vec_phi)
    
    #-----------Landmarks
    land_mx = state_vector[3+2*landmark_id]
    land_my = state_vector[4+2*landmark_id]
    
    #---------- Difference between landmark and robot position
    delta_x = land_mx - state_vec_x
    delta_y = land_my - state_vec_y
    
    #---------- Distance calculation
    laser_range = np.sqrt(delta_x**2 + delta_y**2)
    laser_bearing = slam_utils.clamp_angle(np.arctan2(delta_y,delta_x)-state_vec_phi)
    
    #---------Measurements(calculated)
    zhat = [[laser_range],[laser_bearing]]
    size = len(state_vector)
    
    h_1 = [-laser_range * delta_x,-laser_range * delta_y,0,laser_range * delta_x, laser_range * delta_y]
    h_2 = [delta_y,-delta_x,-laser_range**2,-delta_y,delta_x]
    obs_h = np.array([h_1 , h_2])/laser_range**2
    
    #-------------Jacobian of state transition matrix
    G = np.zeros((5, size))
    G [:3,:3] = np.eye(3)
    G[3, 3 + 2 * landmark_id] = 1
    G[4, 4 + 2 * landmark_id] = 1
    H = obs_h @ G

    return zhat, H


def initialize_landmark(ekf_state, tree):
    
    # ----------Defining Car Poses
    x = ekf_state['x'][0]
    y = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    
    #----------Defining Tree Measurements
    r = tree[0]
    theta = tree[1]
    
    #----------Defining Tree Coordinates
    x_tree = x + r*np.cos(theta + phi)
    y_tree = y + r*np.sin(theta + phi)
    
    #----------Initializing New Landmark in States
    ekf_state['x'] = np.hstack((ekf_state['x'], x_tree, y_tree))
    
    # ----------Initializing New Covariance
    new_covariance_dimension =  ekf_state['x'].size
    temp_covariance = np.zeros([new_covariance_dimension, new_covariance_dimension])
    temp_covariance[0:new_covariance_dimension-2, 0:new_covariance_dimension-2] = ekf_state['P']
    temp_covariance[new_covariance_dimension-2, new_covariance_dimension-2] = 1000
    temp_covariance[new_covariance_dimension-1, new_covariance_dimension-1] = 1000
    
    # ----------New Covariance
    ekf_state['P'] = temp_covariance
    
    # ----------Incrementing Landmark ID
    ekf_state['num_landmarks'] = ekf_state['num_landmarks'] + 1
    
    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    
    #-----------Initializing new landmark
    if ekf_state["num_landmarks"] == 0:
        return [-1 for m in measurements]
    
    #-----------CHI2 Distribution Parameters
    alpha = chi2.ppf(0.95, df=2)
    beta = chi2.ppf(0.999, df=2)

    n_lmark = ekf_state['num_landmarks']
    n_scans = len(measurements)
    Q = np.diag(np.array([sigmas['range']**2, sigmas['bearing']**2]))
    
    #-----------Cost Matrix M
    M = alpha * np.ones((n_scans, n_lmark + n_scans))

    for j in range(n_lmark):
        zhat_j, H = laser_measurement_model(ekf_state, j)
        S = (H @ ekf_state['P'] @ H.T) + Q
        for i in range(n_scans):
            residuals = measurements[i][:2] - np.squeeze(zhat_j)  
            mahalanobis_dist = residuals.T @ slam_utils.invert_2x2_matrix(S) @ residuals
            M[i, j] = mahalanobis_dist

    #-----------Finding Pairs for measurement-landmark correspondence
    matches = slam_utils.solve_cost_matrix_heuristic(np.copy(M))
    matches.sort() 
    associations = list(range(n_scans))

    #-----------Disregarding associations that are too ambiguous
    for i in range(n_scans):
        scan_index = matches[i][0]
        landmark_index = matches[i][1]
        if landmark_index >= n_lmark:
            if np.amin(M[scan_index, :n_lmark]) > beta:
                associations[scan_index] = -1
            else:
                associations[scan_index] = -2
        else:
            associations[scan_index] = landmark_index

    return associations

def laser_update(trees, assoc, ekf_state, sigmas, params):
    
    #----------Measurement Noise 'Q'
    Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])
    
    #----------Initiliazing Landmark and EKF Update
    for i in range(len(trees)):
        j = assoc[i]
        if j == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            j = int(len(ekf_state['x'])/2) - 2
        elif j == -2:
            continue
        dimension = ekf_state['x'].shape[0]
        states_bar = ekf_state['x']
        covariance_bar = ekf_state['P']
        z_hat, H = laser_measurement_model(ekf_state, j)
        S = (H @ covariance_bar @ H.T) + Q
        S_inv = np.linalg.inv(S)
        K = covariance_bar @ H.T @ S_inv
        z = np.array([
                      [trees[i][0]],
                      [trees[i][1]]
                      ])
        delta = z - z_hat
        states = states_bar + np.squeeze(K @ delta)
        states[2] = slam_utils.clamp_angle(states[2])
        ekf_state['x'] = states
        covariance = (np.eye(dimension) - (K @ H)) @ covariance_bar
        covariance = slam_utils.make_symmetric(covariance)
        ekf_state['P'] = covariance
        
    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    
    #----------Defining EKF states
    ekf_state = {
                 'x': ekf_state_0['x'],
                 'P': ekf_state_0['P'],
                 'num_landmarks': ekf_state_0['num_landmarks']
                 }
    
    #----------Defining state history
    state_history = {
                     't': [0],
                     'x': ekf_state['x'],
                     'P': np.diag(ekf_state['P'])
                     }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()
        
    lidar_data = []    
    gps_time = []
    gps_ground_truth = []
    gps_ekf_state = []
    gps_ekf_covariance = []
    for i, event in enumerate(events):
        print(i)
        t = event[1][0]
        
        #----------GPS Event
        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)
            lidar_data.append(np.zeros(361))
            gps_time.append(event[1][0])
            gps_ground_truth.append(gps_msmt)
            gps_ekf_state.append(ekf_state['x'][:2])
            gps_ekf_covariance.append(ekf_state['P'][:2,:2])
            
        #----------Odometry Event
        elif event[0] == 'odo':
            lidar_data.append(np.zeros(361))
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t
            
        #----------Laser Event
        elif event[0] == 'laser':
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)
            lidar_data.append(scan)

        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history, lidar_data, gps_time, gps_ground_truth, gps_ekf_state, gps_ekf_covariance

def writing_data_for_error(gps_time, gps_ground_truth, gps_ekf_state, gps_ekf_covariance):
    with open('GPS Ground Truth & Estimated Data.txt', 'w') as f:
        for val0, val1, val2, val3 in zip(gps_time, gps_ground_truth, gps_ekf_state, gps_ekf_covariance):
            f.write(str(val0))
            f.write(', ')
            f.write(str(val1[0]))
            f.write(', ')
            f.write(str(val1[1]))
            f.write(', ')
            f.write(str(val2[0]))
            f.write(', ')
            f.write(str(val2[1]))
            f.write(', ')
            f.write(str(val3[0,0]))
            f.write(', ')
            f.write(str(val3[0,1]))
            f.write(', ')
            f.write(str(val3[1,0]))
            f.write(', ')
            f.write(str(val3[1,1]))
            f.write("\n")

def error_calculation():
    temp = slam_utils.read_data_file("GPS Ground Truth & Estimated Data.txt")
    gps_time = []
    gps_ground_truth = []
    gps_ekf_state = []
    gps_ekf_covariance = []
    for line in temp:
        gps_time.append(line[0])
        gps_ground_truth.append(line[1:3])
        gps_ekf_state.append(line[3:5])
        P11 = line[5]; P12 = line[6]; P21 = line[7]; P22 = line[8] 
        temp_cov = np.array([
                            [P11, P12],
                            [P21, P22]
                            ])
        gps_ekf_covariance.append(temp_cov)
    
    
    dim = len(gps_ekf_covariance)
    sigma_x = list()
    sigma_y = list()
    
    for i in range(dim):
        sigma_x.append(np.sqrt(gps_ekf_covariance[i][0,0]))
        sigma_y.append(np.sqrt(gps_ekf_covariance[i][1,1]))
    
    x_error = [gps_ekf_state[i][0] - gps_ground_truth[i][0] for i in range(dim)] 
    y_error = [gps_ekf_state[i][1] - gps_ground_truth[i][1] for i in range(dim)]
    
    plt.figure()
    plt.plot(gps_time, x_error, color='blue', label='x-diff')
    plt.plot(gps_time, [3 * sigma for sigma in sigma_x], color='red', label='+3-sigma')
    plt.plot(gps_time, [-3 * sigma for sigma in sigma_x], color='red', label='-3-sigma')
    plt.xlabel('Number of Steps')
    plt.ylabel('Pose Error')
    plt.title('X Error')
    plt.legend(loc='lower left')
    plt.grid()
    
    plt.figure()
    plt.plot(gps_time, y_error, color='blue', label='y-diff')
    plt.plot(gps_time, [3 * sigma for sigma in sigma_y], color='red', label='+3-sigma')
    plt.plot(gps_time, [-3 * sigma for sigma in sigma_y], color='red', label='-3-sigma')
    plt.xlabel('Number of Steps')
    plt.ylabel('Pose Error')
    plt.title('Y Error')
    plt.legend(loc='lower left')
    plt.grid()
    
    x_truth = []
    x_cal = []
    for val1, val2 in zip(gps_ground_truth, gps_ekf_state):
        x_truth.append(val1[0])
        x_cal.append(val2[0])
        
    plt.figure()
    plt.plot(gps_time, x_truth, label = 'true data')
    plt.plot(gps_time, x_cal, label = 'ekf data')
    plt.xlabel('Time (sec)')
    plt.ylabel('X - coordinate (m)')
    plt.title('Ground Truth vs EKF Estimates')
    plt.legend()
    plt.grid()
    
    y_truth = []
    y_cal = []
    for val1, val2 in zip(gps_ground_truth, gps_ekf_state):
        y_truth.append(val1[1])
        y_cal.append(val2[1])
        
    plt.figure()
    plt.plot(gps_time, y_truth, label = 'true data')
    plt.plot(gps_time, y_cal, label = 'ekf data')
    plt.xlabel('Time (sec)')
    plt.ylabel('Y - coordinate (m)')
    plt.title('Ground Truth vs EKF Estimates')
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.plot([x_true[0] for x_true in gps_ground_truth], [y_true[1] for y_true in gps_ground_truth], color='red', label='true trajectory')
    plt.plot([x_ekf[0] for x_ekf in gps_ekf_state], [y_ekf[1] for y_ekf in gps_ekf_state], color='blue', label='EKF trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True vs EKF Trajectory')
    plt.legend(loc='lower left')
    plt.grid()
    
    
def writing_results_file(state_history, lidar_data):
    
    #----------Writing the results to an output text file
    count = 0
    with open('Extracted States & Lidar Data.txt', 'w') as f:
        for result, LD in zip(state_history['x'], lidar_data):
            print("writing data: ", count)
            x = result[0]
            y = result[1]
            phi = result[2]
            f.write("{:>4}".format(x))
            f.write(', ')
            f.write("{:>4}".format(y))
            f.write(', ')
            f.write("{:>4}".format(phi))
            for val in LD:
                f.write(', ')
                f.write("{:>4}".format(val))
            f.write("\n")
            count = count + 1

def main():
    
    #----------Reading data files
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    #----------Compiling all events
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    #----------Vehicle Parameters
    vehicle_params = {
                      "a": 3.78,
                      "b": 0.50,
                      "L": 2.83,
                      "H": 0.76
                      }
    
    #----------Measurement Parameters
    filter_params = {
                    "max_laser_range": 75, # meters
                    "do_plot": False,
                    "plot_raw_laser": False,
                    "plot_map_covariances": False
                    }

    #----------Noise values
    sigmas = {
              #----------Motion noise
              "xy": 0.05,
              "phi": 0.5*np.pi/180,
              #----------Measurement noise
              "gps": 3,
              "range": 0.5,
              "bearing": 5*np.pi/180
              }

    #----------Initial state of filter
    ekf_state = {
                 "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
                 "P": np.diag([.1, .1, 1]),
                 "num_landmarks": 0
                 }

    state_history, lidar_data, gps_time, gps_ground_truth, gps_ekf_state, gps_ekf_covariance = run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)
    writing_data_for_error(gps_time, gps_ground_truth, gps_ekf_state, gps_ekf_covariance)
    error_calculation()
    writing_results_file(state_history, lidar_data)

if __name__ == '__main__':
    main()

