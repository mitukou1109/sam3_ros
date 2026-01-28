# sam3_ros

🤖 ROS 2 wrapper for [SAM 3](https://github.com/facebookresearch/sam3)

## 📋 Prerequisites

- Python 3.10+
- PyTorch 2.7+
- CUDA-compatible GPU (CUDA 12.6+)
- ROS 2 Humble or later

## 🚀 Installation

### 1. Clone the Repository

```bash
cd ~/ros2_ws/src
git clone https://github.com/mitukou1109/sam3_ros.git
```

### 2. Install Dependencies

```bash
cd ~/ros2_ws/src
rosdep install -iyr --from-paths .
```

### 3. Build the Workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

## 🔑 Authentication

> [!IMPORTANT]
> SAM 3 checkpoints require access approval. Follow these steps:
>
> 1. Request access on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3)
> 2. After approval, create a Read token in your [Hugging Face account settings](https://huggingface.co/settings/tokens)
> 3. Login using your token:
>
> ```bash
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

TBD

## ⚙️ Parameters

| Parameter              | Type     | Description                                                                             | Default |
| ---------------------- | -------- | --------------------------------------------------------------------------------------- | ------- |
| `use_compressed_image` | `bool`   | Use compressed image input. If `true`, subscribes to `/image/compressed`, else `/image` | `true`  |
| `text_prompt`          | `string` | Text prompt for guided segmentation                                                     | `""`    |
| `confidence_threshold` | `float`  | Minimum confidence threshold for mask filtering                                         | `0.5`   |
