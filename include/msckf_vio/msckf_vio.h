/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <msckf_vio/CameraMeasurement.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_srvs/Trigger.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cam_state.h"
#include "feature.hpp"
#include "imu_state.h"

namespace msckf_vio {
/**
 * @brief MsckfVio
 * Implements the algorithm in Anatasios I. Mourikis, and Stergios I. Roumeliotis, "A Multi-State Constraint Kalman
 * Filter for Vision-aided Inertial Navigation", http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class MsckfVio {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Constructor
  MsckfVio(ros::NodeHandle& pnh);
  /// 防止浅拷贝
  MsckfVio(const MsckfVio&) = delete;
  MsckfVio operator=(const MsckfVio&) = delete;

  /// Destructor
  ~MsckfVio() {}

  /**
   * @brief initialize
   * 初始化VIO.
   */
  bool initialize();

  /**
   * @brief reset
   * 将VIO重置为初始状态
   */
  void reset();

  typedef boost::shared_ptr<MsckfVio> Ptr;
  typedef boost::shared_ptr<const MsckfVio> ConstPtr;

 private:
  /**
   * @brief StateServer
   * 存储一个IMU状态和几个相机状态，用于构建测量模型.
   */
  struct StateServer {
    IMUState imu_state;
    CamStateServer cam_states;

    /// 状态协方差矩阵
    Eigen::MatrixXd state_cov;
    Eigen::Matrix<double, 12, 12> continuous_noise_cov;  // 连续噪声协方差
  };

  /**
   * @brief loadParameters
   * 加载配置参数
   */
  bool loadParameters();

  /**
   * @brief createRosIO
   * Create ros publisher and subscirbers.
   */
  bool createRosIO();

  /**
   * @brief imuCallback
   * Callback function for the imu message.
   * @param msg IMU msg.
   */
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  /**
   * @brief featureCallback
   * Callback function for feature measurements.
   * @param msg Stereo feature measurements.
   */
  void featureCallback(const CameraMeasurementConstPtr& msg);

  /**
   * @brief publish
   * Publish the results of VIO.
   * @param time The time stamp of output msgs.
   */
  void publish(const ros::Time& time);

  /**
   * @brief initializegravityAndBias
   * 根据前几个IMU读数初始化IMU bias和初始方向。
   */
  void initializeGravityAndBias();

  /**
   * @biref resetCallback
   * Callback function for the reset service. Note that this is NOT anytime-reset. This function should only be called
   * before the sensor suite starts moving. e.g. while the robot is still on the ground.
   */
  bool resetCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

  /// Filter related functions
  /// Propogate the state
  void batchImuProcessing(const double& time_bound);
  void processModel(const double& time, const Eigen::Vector3d& m_gyro, const Eigen::Vector3d& m_acc);
  void predictNewState(const double& dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc);

  /// Measurement update
  void stateAugmentation(const double& time);

  ///@brief 增加测量观测
  void addFeatureObservations(const CameraMeasurementConstPtr& msg);

  ///@brief 用于计算在单个相机帧处观察到的单个特征的测量雅可比.
  void measurementJacobian(const StateIDType& cam_state_id,
                           const FeatureIDType& feature_id,
                           Eigen::Matrix<double, 4, 6>& H_x,
                           Eigen::Matrix<double, 4, 3>& H_f,
                           Eigen::Vector4d& r);
  ///@brief 计算在该特征在给定相机状态下观察到的所有测量的雅可比
  void featureJacobian(const FeatureIDType& feature_id,
                       const std::vector<StateIDType>& cam_state_ids,
                       Eigen::MatrixXd& H_x,
                       Eigen::VectorXd& r);
  void measurementUpdate(const Eigen::MatrixXd& H, const Eigen::VectorXd& r);

  bool gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const int& dof);

  void removeLostFeatures();

  void findRedundantCamStates(std::vector<StateIDType>& rm_cam_state_ids);

  void pruneCamStateBuffer();

  ///@brief 如果不确定性太大,则将系统重置
  void onlineReset();

  /// 卡方检验表
  static std::map<int, double> chi_squared_test_table;

  /// State vector
  StateServer state_server;

  int max_cam_state_size;  // 存储的最多相机状态量数目

  /// Features used
  MapServer map_server;

  /// IMU data buffer
  /// This is buffer is used to handle the unsynchronization or transfer delay between IMU and Image messages.
  std::vector<sensor_msgs::Imu> imu_msg_buffer;

  bool is_gravity_set;   // 重力矢量设置标志位
  bool is_first_img;     // 首帧图像标志位
  double tracking_rate;  // 跟踪频率

  /// The position uncertainty threshold is used to determine when to reset the system online.
  /// Otherwise, the ever-increaseing uncertainty will make the estimation unstable.
  /// Note this online reset will be some dead-reckoning.
  /// Set this threshold to nonpositive to disable online reset.
  double position_std_threshold;

  /// Threshold for determine keyframes
  double translation_threshold;
  double rotation_threshold;
  double tracking_rate_threshold;

  /// Ros node handle
  ros::NodeHandle nh;

  /// Subscribers and publishers
  ros::Subscriber imu_sub;
  ros::Subscriber feature_sub;
  ros::Publisher odom_pub;
  ros::Publisher feature_pub;
  tf::TransformBroadcaster tf_pub;
  ros::ServiceServer reset_srv;

  /// Frame id
  std::string fixed_frame_id;
  std::string child_frame_id;

  /// Whether to publish tf or not.
  bool publish_tf;

  /// Frame rate of the stereo images.
  /// This variable is only used to determine the timing threshold of each iteration of the filter.
  double frame_rate;

  /// Debugging variables and functions
  void mocapOdomCallback(const nav_msgs::OdometryConstPtr& msg);

  ros::Subscriber mocap_odom_sub;
  ros::Publisher mocap_odom_pub;
  geometry_msgs::TransformStamped raw_mocap_odom_msg;
  Eigen::Isometry3d mocap_initial_frame;
};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

}  // namespace msckf_vio

#endif
