#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
/// CHECK: include needed ROS msg type headers and libraries

class ReactiveFollowGap : public rclcpp::Node {
// Implement Reactive Follow Gap on the car
// This is just a template, you are free to implement your own node!

public:
    ReactiveFollowGap() : Node("reactive_node") 
    {
        /// TODO: create ROS subscribers and publishers
        lidar_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic, 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));
        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 100);
    }

private:

    float lidar_max = 5.0;
    float dist_threshold = 3.0;
    float cliff_threshold = 1.0;
    int cliff_extension = 10;
    int rb = 5;

    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";

    /// TODO: create ROS subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;

    void preprocess_lidar(std::vector<float> &ranges) 
    {
        // Preprocess the LiDAR scan array. Expert implementation includes:
        // 1.Setting each value to the mean over some window
        // 2.Rejecting high values (eg. > 3m)

        std::replace_if(ranges.begin(), ranges.end(), [this](float range) {
            return std::isinf(range) || range > lidar_max;
        }, lidar_max);

        for (size_t i = 0; i < ranges.size(); ++i) {
            int count = 0;
            float sum = 0.0;
            for (int j = -rb; j <= rb; ++j) {
                if ((i + j) >= 0 && (i + j) < ranges.size()) {
                    sum += ranges[i + j];
                    count++;
                }
            }
            ranges[i] = sum / count;
        }
    }

    std::pair<int, int> find_max_gap(const std::vector<float> &ranges) 
    {
        // Return the start index & end index of the max gap in free_space_ranges
        int max_gap_start = -1;
        int max_gap_end = -1;
        int max_gap_length = 0;
        int current_gap_start = -1;
        int current_gap_length = 0;

        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i] > dist_threshold) {
                if (current_gap_start < 0) current_gap_start = i;
                current_gap_length++;
            } else {
                if (current_gap_length > max_gap_length) {
                    max_gap_length = current_gap_length;
                    max_gap_start = current_gap_start;
                    max_gap_end = i - 1;
                }
                current_gap_start = -1;
                current_gap_length = 0;
            }
        }

        if (current_gap_length > max_gap_length) {
            max_gap_length = current_gap_length;
            max_gap_start = current_gap_start;
            max_gap_end = ranges.size() - 1;
        }

        return std::make_pair(max_gap_start, max_gap_end);
    }

    int find_best_point(const std::vector<float> &ranges, int start_i, int end_i) 
    {
        // Start_i & end_i are start and end indicies of max-gap range, respectively
        // Return index of best point in ranges
	    // Naive: Choose the furthest point within ranges and go there
        auto max_iter = std::max_element(ranges.begin() + start_i, ranges.begin() + end_i + 1);
        return std::distance(ranges.begin(), max_iter);
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) 
    {
        // Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        auto ranges = scan_msg->ranges;
        preprocess_lidar(ranges);

        /// TODO:
        // Find closest point to LiDAR
        int closest_point = std::distance(ranges.begin(), std::min_element(ranges.begin(), ranges.end()));
        std::fill(ranges.begin() + std::max(closest_point - rb, 0), ranges.begin() + std::min(closest_point + rb + 1, (int)ranges.size()), 0.0);
        
        // Eliminate all points inside 'bubble' (set them to zero) 
        auto [start_i, end_i] = find_max_gap(ranges);

        // Find max length gap
        int best_point = find_best_point(ranges, start_i, end_i);
        float angle = scan_msg->angle_min + best_point * scan_msg->angle_increment;

        // Find the best point in the gap 

        // Publish Drive message
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->get_clock()->now();
        drive_msg.drive.speed = 0.5; // set a constant speed
        drive_msg.drive.steering_angle = angle;
    
        drive_pub->publish(drive_msg);
    }
};

int main(int argc, char ** argv) {
rclcpp::init(argc, argv);
auto node = std::make_shared<ReactiveFollowGap>();
rclcpp::spin(node);
rclcpp::shutdown();
return 0;
}
