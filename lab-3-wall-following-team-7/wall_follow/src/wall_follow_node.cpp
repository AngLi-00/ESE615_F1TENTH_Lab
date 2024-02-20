#include "rclcpp/rclcpp.hpp"
#include <string>
#include <cmath> 
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class WallFollow : public rclcpp::Node {
public:
    WallFollow() : Node("wall_follow_node")
    {
        // TODO: create ROS subscribers and publishers
        scan_subscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic, 10, std::bind(&WallFollow::scan_callback, this, std::placeholders::_1));
        
        drive_publisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            drive_topic, 10);
    }

private:
    // PID CONTROL PARAMS
    double kp = 4.5; 
    double ki = 0.05; 
    double kd = 4.3; 
    double servo_offset = 0.0;
    double prev_error = 0.0;
    double error = 0.0;
    double integral = 0.0;
    double L = 1; 
    double L_Wall = 1;
    double prev_time = 0.0;
    double angle = 0.0;
    double alpha = 0.0;
    double time_gap = 0.0;

    // Topics
    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";
    // TODO: create ROS subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher;
    
    // Method to get the range from a scan at a given angle
    double get_range(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg, double angle)
    {
        // Using size_type to avoid signed/unsigned mismatch
        std::vector<int>::size_type index = static_cast<std::vector<int>::size_type>((angle - scan_msg->angle_min) / scan_msg->angle_increment);
        
        double range_measurement = scan_msg->ranges[index];

        // Retrieve the range value at the computed index, handling NaNs and infs
        if (std::isnan(range_measurement) || std::isinf(range_measurement)) {
        range_measurement = scan_msg->range_max;
        }

        return range_measurement;
    }


    // Method to calculate the error based on the current distance to the wall
    double get_error(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg, double dist)
    {
        double theta = 45 * M_PI / 180.0; 
        double a = get_range(scan_msg, M_PI_2 - theta); // Range at 90 degrees to the right
        double b = get_range(scan_msg, M_PI_2); // Range at 'theta' degrees to the right

        alpha = atan2(a * cos(theta) - b, a * sin(theta)); // Angle of the car relative to the wall
        double Dt = b * cos(alpha); // Current distance to the wall
        double Dt_future = Dt + L * sin(alpha); // Predicted future distance to the wall
        double error = dist - Dt_future; 

        return error;
    }

    void pid_control(double error, double velocity) 
    { 
        if (time_gap > 0) 
        {
        // Calculate integral of error
        integral += error * time_gap;
        // Calculate P, I, D terms
        double P = kp * error; // Proportional term
        double I = ki * integral; // Integral term
        double D = kd * (error - prev_error) / time_gap; // Derivative term
        // PID formula to calculate steering angle
        angle = P + I + D;
        }

        // Update previous error and time for the next call
        prev_error = error;
        // Populate the drive message
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.speed = velocity;
        drive_msg.drive.steering_angle = -angle;
        // Publish the drive message
        drive_publisher->publish(drive_msg);
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        double error = get_error(scan_msg, L_Wall);
        double velocity = 0.0;
        if (std::abs(alpha) <= M_PI / 18) { 
            velocity = 1.5; 
        }
        else if (std::abs(alpha) > M_PI / 18 && std::abs(alpha) <= M_PI / 9) { 
            velocity = 1.0; 
        } 
        else { 
            velocity = 0.5; 
        }

        pid_control(error, velocity);

        double current_time = this->now().seconds();
        time_gap = prev_time > 0 ? current_time - prev_time : 0.0;
        prev_time = current_time;
        // RCLCPP_INFO(this->get_logger(), "PID Control");
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WallFollow>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}