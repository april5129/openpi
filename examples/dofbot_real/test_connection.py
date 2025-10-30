#!/usr/bin/env python3
"""
测试与远程pi0.5服务器的连接
"""

import numpy as np
from openpi_client import websocket_client_policy


def test_server_connection():
    """测试服务器连接和基本通信"""
    server_host = "12.148.158.61"
    server_port = 8000
    
    print(f"测试连接到: {server_host}:{server_port}")
    
    try:
        # 使用官方openpi_client库连接
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_host,
            port=server_port
        )
        
        print("✅ 连接成功!")
        print(f"📋 服务器元数据: {policy.get_server_metadata()}")
        
        # 发送一个DROID格式的测试观测
        test_obs = {
            "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": "test connection"
        }
        
        print("📤 发送测试数据...")
        print("📋 观测数据结构:")
        for key, value in test_obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"  {key}: {value}")
        
        action_data = policy.infer(test_obs)
        
        print("✅ 收到响应:")
        print(f"📋 完整响应数据: {action_data}")
        
        # 检查响应中的所有键
        print("📋 响应中的所有键:")
        for key in action_data.keys():
            value = action_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        # 检查不同可能的动作字段
        if 'actions' in action_data:
            actions = action_data['actions']
            print(f"📋 动作数据 (actions): shape={actions.shape}")
            print(f"  第一个动作: {actions[0] if len(actions) > 0 else 'None'}")
        elif 'robot_action' in action_data:
            robot_action = action_data['robot_action']
            print(f"📋 动作数据 (robot_action): {robot_action}")
        else:
            print("❌ 未找到动作数据字段!")
        
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


if __name__ == "__main__":
    success = test_server_connection()
    if success:
        print("\n🎉 服务器连接测试成功! 可以运行主程序了.")
    else:
        print("\n💥 服务器连接失败! 请检查:")
        print("  1. 服务器是否在 12.148.158.61:8000 运行")
        print("  2. 网络连接是否正常")
        print("  3. 防火墙设置")