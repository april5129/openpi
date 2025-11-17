#!/usr/bin/env python3
"""
简化版客户端: 只使用图像和提示与服务器交互一轮
所有机械臂相关信息设为0
"""

import numpy as np
import cv2
import os
from openpi_client import websocket_client_policy


class SimpleClient:
    """极简WebSocket客户端 - 只发送图像和提示"""
    
    def __init__(self, server_host="127.0.0.1", server_port=8000):
        # 连接到远程pi0.5服务器
        print(f"🔌 连接到远程服务器: {server_host}:{server_port}")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_host,
            port=server_port
        )
        print(f"✅ 连接成功! 服务器元数据: {self.policy.get_server_metadata()}")
    
    def load_image(self, image_path):
        """加载图像文件"""
        if not os.path.exists(image_path):
            print(f"⚠️  图像文件不存在: {image_path}")
            # 返回黑色图像
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"⚠️  无法读取图像: {image_path}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 调整大小并转换颜色空间
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        print(f"📷 成功加载图像: {image_path}")
        print(f"   - 图像形状: {frame.shape}")
        print(f"   - 图像数据类型: {frame.dtype}")
        
        return frame
    
    def get_observation(self, image_path, prompt="pick up the object"):
        """构建观测数据 - 图像+提示，机械臂信息全为0"""
        # 加载图像
        frame = self.load_image(image_path)
        
        # 构建观测 - 按DROID格式，所有关节信息设为0
        joint_positions = np.zeros(7, dtype=np.float32)  # 7个关节全为0
        gripper_position = np.array([0.0], dtype=np.float32)  # 夹爪位置为0
        
        obs = {
            "observation/exterior_image_1_left": frame,  # numpy数组
            "observation/wrist_image_left": frame,  # 用同一个图像
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_position,
            "prompt": prompt
        }
        
        # 打印发送的观测数据
        print("\n" + "="*60)
        print("📤 发送给服务器的观测数据:")
        print("="*60)
        print(f"🖼️  图像信息:")
        print(f"   - 图像形状: {frame.shape}")
        print(f"   - 图像数据类型: {frame.dtype}")
        print(f"📐 关节位置: {joint_positions} (全为0)")
        print(f"🤏 夹爪位置: {gripper_position[0]:.3f} (设为0)")
        print(f"💬 任务提示: '{prompt}'")
        print("="*60)
        
        return obs
    
    def run_once(self, image_path, prompt="pick up the object"):
        """只执行一轮交互"""
        print(f"\n🚀 开始单轮交互")
        print(f"   - 图像路径: {image_path}")
        print(f"   - 任务提示: '{prompt}'")
        
        try:
            # 1. 获取观测数据
            obs = self.get_observation(image_path, prompt)
            
            # 2. 发送到服务器并接收响应
            print("\n📡 正在发送观测数据到服务器...")
            import time
            start_time = time.time()
            action_data = self.policy.infer(obs)
            inference_time = time.time() - start_time
            
            # 3. 显示服务器响应
            print("\n📥 收到服务器响应:")
            print(f"   - 响应类型: {type(action_data)}")
            print(f"   - 网络往返时间: {inference_time:.3f}s")
            
            if isinstance(action_data, dict):
                # 显示响应内容
                print(f"   - 响应字段: {list(action_data.keys())}")
                
                # 显示动作序列信息
                if "actions" in action_data:
                    actions = action_data["actions"]
                    print(f"\n📊 动作序列信息:")
                    print(f"   - 动作序列长度: {len(actions)}")
                    if len(actions) > 0:
                        print(f"   - 首个动作维度: {len(actions[0])}")
                        print(f"   - 首个动作: {actions[0]}")
                        print(f"   - 首个动作范围: [{min(actions[0]):.3f}, {max(actions[0]):.3f}]")
                
                # 显示时序信息
                if "policy_timing" in action_data:
                    print(f"\n⏱️  策略推理时间: {action_data['policy_timing']}")
                if "server_timing" in action_data:
                    print(f"⏱️  服务器时序: {action_data['server_timing']}")
            
            print("\n✅ 单轮交互完成")
            return action_data
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版pi0.5客户端 - 只交互一轮")
    parser.add_argument("--host", default="127.0.0.1", help="服务器IP")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--image", default="/root/ljw/openpi/examples/dofbot_real/test.jpg", help="图像文件路径")
    parser.add_argument("--prompt", default="pick up the object", help="任务描述")
    
    args = parser.parse_args()
    
    print("🤖 简化版pi0.5客户端 - 单轮交互模式")
    print(f"服务器: {args.host}:{args.port}")
    print(f"图像: {args.image}")
    print(f"任务: {args.prompt}")
    
    # 创建客户端并执行单轮交互
    client = SimpleClient(args.host, args.port)
    result = client.run_once(args.image, args.prompt)
    
    if result:
        print("\n🎉 交互成功完成!")
    else:
        print("\n😞 交互失败")


if __name__ == "__main__":
    main()

