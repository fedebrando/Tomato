from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    params = {
        "ref_link": "camera_color_optical_frame",
        "camera_info_topic": "/camera/camera/color/camera_info",
        "pcl_topic": "converted_pcl",
        "rgb_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    }

    node = Node(
        package="images_to_pcl_converter",
        executable="images_to_pcl_converter",
        output="screen",
        parameters=[
            params,
        ],
    )

    return LaunchDescription([node])
