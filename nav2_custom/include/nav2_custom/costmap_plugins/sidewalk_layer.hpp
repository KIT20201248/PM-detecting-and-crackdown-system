#pragma once

#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>

namespace nav2_custom {

class SidewalkLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SidewalkLayer();
  virtual void onInitialize() override;
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double *min_x, double *min_y, double *max_x, double *max_y) override;
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D& master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  virtual void reset() override;

  bool isClearable() override { return false; }

private:
  void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
  bool received_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_;
};

}  // namespace nav2_custom
