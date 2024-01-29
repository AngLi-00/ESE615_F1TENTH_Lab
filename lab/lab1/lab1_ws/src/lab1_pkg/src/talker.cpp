// Criteria:
// @ talker listens to two ROS parameters v and d.
// @ talker publishes an AckermannDriveStamped message
// with the speed field equal to the v parameter and
// steering_angle field equal to the d parameter, and
// to a topic named drive.
// @ talker publishes as fast as possible.
// @ To test node, set the two ROS parameters through
// command line, a launch file, or a yaml file.

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;  // Introduce some time units into the
                                       // current scope

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class Talker : public rclcpp::Node {
 public:
  Talker() : Node("talker") {
    this->declare_parameter<float>("v", 0.0);
    this->declare_parameter<float>("d", 0.0);

    publisher_ =
        this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "drive", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(10),
                                     std::bind(&Talker::timer_callback, this));
  }

 private:
  void
  timer_callback() {  // The timer_callback function is where the message data
                      // is set and the messages are actually published.
    auto message = ackermann_msgs::msg::AckermannDriveStamped();
    double v, d;
    this->get_parameter("v", v);
    this->get_parameter("d", d);

    message.drive.speed = v;
    message.drive.steering_angle = d;

    publisher_->publish(message);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
      publisher_;
};

// The main function is where the node actually executes.
// rclcpp::init initializes ROS 2
// rclcpp::spin starts processing data from the node, including callbacks from
// the timer.

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Talker>());
  rclcpp::shutdown();
  return 0;
}