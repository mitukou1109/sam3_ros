# sam3_ros

🤖 ROS 2 wrapper for [SAM 3](https://github.com/facebookresearch/sam3)

## 📋 Prerequisites

- Python 3.10+
- PyTorch 2.7+
- CUDA-compatible GPU (CUDA 12.6+)
- ROS 2 Humble+

## 🚀 Installation

### 1. Clone

```bash
cd ~/ros2_ws/src
git clone https://github.com/mitukou1109/sam3_ros.git
```

### 2. Install dependencies

```bash
cd ~/ros2_ws/src
rosdep install -iyr --from-paths .
```

### 3. Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

## 🔑 Authentication

> [!IMPORTANT]
> SAM 3 checkpoints require access approval:
>
> 1. Request access on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3)
> 2. Create a **Read** token in your [Hugging Face settings](https://huggingface.co/settings/tokens)
> 3. Login with your token:
>
> ```bash
> cd ~/ros2_ws/src/sam3_ros
> uv run hf auth login
> # Enter your token (input will not be visible): [paste your token]
> ```

## 💻 Usage

```bash
source ~/ros2_ws/install/local_setup.bash
ros2 run sam3_ros segmentation_node
```

## 📡 Topics

### Subscribed

| Topic               | Type                              | Description            |
| ------------------- | --------------------------------- | ---------------------- |
| `/image`            | `sensor_msgs/msg/Image`           | Raw input image        |
| `/image/compressed` | `sensor_msgs/msg/CompressedImage` | Compressed input image |

### Published

| Topic                       | Type                               | Description                                     |
| --------------------------- | ---------------------------------- | ----------------------------------------------- |
| `~/mask_image/compressed`   | `sensor_msgs/msg/CompressedImage`  | Mask image (pixel value corresponds to mask ID) |
| `~/detections`              | `vision_msgs/msg/Detection2DArray` | Bounding boxes of detected masks                |
| `~/result_image/compressed` | `sensor_msgs/msg/CompressedImage`  | Image with colored masks for visualization      |

## ⚙️ Parameters

| Parameter                   | Type     | Description                                                                             | Default |
| --------------------------- | -------- | --------------------------------------------------------------------------------------- | ------- |
| `use_compressed_image`      | `bool`   | Use compressed image input. If `true`, subscribes to `/image/compressed`, else `/image` | `true`  |
| `text_prompt`               | `string` | Text prompt for guided segmentation                                                     | `""`    |
| `confidence_threshold`      | `float`  | Minimum confidence threshold for mask filtering                                         | `0.5`   |
| `result_visualization_rate` | `float`  | Publish rate (Hz) for visualization results                                             | `10.0`  |
