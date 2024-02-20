#include "rclcpp/rclcpp.hpp"
/// CHECK: include needed ROS msg type headers and libraries
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"


class Safety : public rclcpp::Node {
// The class that handles emergency braking

public:
    Safety() : Node("safety_node")
    {
        /*
        You should also subscribe to the /scan topic to get the
        sensor_msgs/LaserScan messages and the /ego_racecar/odom topic to get
        the nav_msgs/Odometry messages

        The subscribers should use the provided odom_callback and 
        scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        */

        /// TODO: create ROS subscribers and publishers
        
        odom_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&Safety::odom_callback, this, std::placeholders::_1));

        scan_subscriber = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&Safety::scan_callback, this, std::placeholders::_1));

        drive_publisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);
    }

private:
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher;

    double speed = 0.0;

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        /// TODO: update current speed
        speed = msg->twist.twist.linear.x;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        std::vector<double> ittc_values; // store iTTC for each angle

        /// TODO: calculate TTC
        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            double r = scan_msg->ranges[i];
            double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
            double range_rate = -speed * std::cos(angle);

            double ittc = range_rate < 0 ? r / (-range_rate) : std::numeric_limits<double>::infinity();
            ittc_values.push_back(ittc);
        }

        /// TODO: publish drive/brake message
        for (auto ittc : ittc_values) {
            if (ittc < 1.1) {
                ackermann_msgs::msg::AckermannDriveStamped drive_msg;
                drive_msg.drive.speed = 0.0; // “AEB”
                drive_publisher->publish(drive_msg);
                break; // Brake at the first instance of iTTC below threshold
            }
        }
    }


};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}