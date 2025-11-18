#!/usr/bin/env python3
"""
调试脚本：查看实际发送给服务器的观测数据
"""

import cv2
import numpy as np
import os

def check_observation():
    """检查观测数据的图片对应关系"""
    
    print("="*60)
    print("🔍 检查观测数据")
    print("="*60)
    
    # 读取两个摄像头
    print("\n📷 读取摄像头...")
    
    # 摄像头 0 - Microdia USB 2.0 Camera (机械臂上)
    print("  读取摄像头 0 (机械臂上)...")
    cap0 = cv2.VideoCapture(0)
    ret0, frame0 = cap0.read()
    cap0.release()
    
    # 摄像头 2 - Realtek Integrated Webcam (空中全局)
    print("  读取摄像头 2 (空中全局)...")
    cap2 = cv2.VideoCapture(2)
    ret2, frame2 = cap2.read()
    cap2.release()
    
    if not ret0 or not ret2:
        print("❌ 摄像头读取失败！")
        return
    
    print("\n✅ 两个摄像头都读取成功")
    print(f"   摄像头 0 分辨率: {frame0.shape}")
    print(f"   摄像头 2 分辨率: {frame2.shape}")
    
    # 保存原始图片用于对比
    os.makedirs("images", exist_ok=True)
    cv2.imwrite("images/debug_camera0_original.jpg", frame0)
    cv2.imwrite("images/debug_camera2_original.jpg", frame2)
    print("\n💾 已保存原始图片:")
    print("   - images/debug_camera0_original.jpg (机械臂摄像头)")
    print("   - images/debug_camera2_original.jpg (全局摄像头)")
    
    # 模拟客户端的处理流程
    print("\n🔄 模拟客户端处理流程...")
    
    # 处理机械臂摄像头图像 (对应 wrist_image_left)
    wrist_frame = cv2.resize(frame0, (224, 224))
    wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
    
    # 处理全局摄像头图像 (对应 exterior_image_1_left)
    exterior_frame = cv2.resize(frame2, (224, 224))
    exterior_frame_rgb = cv2.cvtColor(exterior_frame, cv2.COLOR_BGR2RGB)
    
    # 保存处理后的图片
    cv2.imwrite("images/debug_wrist_processed.jpg", 
                cv2.cvtColor(wrist_frame_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite("images/debug_exterior_processed.jpg", 
                cv2.cvtColor(exterior_frame_rgb, cv2.COLOR_RGB2BGR))
    
    print("💾 已保存处理后的图片 (224x224, RGB):")
    print("   - images/debug_wrist_processed.jpg (发送为 observation/wrist_image_left)")
    print("   - images/debug_exterior_processed.jpg (发送为 observation/exterior_image_1_left)")
    
    # 显示观测数据结构
    print("\n📋 观测数据结构（发送给服务器的）:")
    print("   1. observation/exterior_image_1_left: 来自摄像头2 (全局视角)")
    print("      └─ 空中俯视，应该能看到整个工作区域")
    print("   2. observation/wrist_image_left: 来自摄像头0 (机械臂视角)")
    print("      └─ 机械臂上的摄像头，近距离看物体")
    
    print("\n" + "="*60)
    print("🤔 问题诊断:")
    print("="*60)
    print("\n可能的问题：")
    print("1. ⚠️  摄像头对应关系是否正确？")
    print("   → 请查看 images/debug_camera*.jpg 确认：")
    print("     - camera0 是否真的在机械臂上？")
    print("     - camera2 是否真的在空中俯视？")
    print("")
    print("2. ⚠️  服务端模型期望的图片顺序可能不对")
    print("   → 尝试交换两个摄像头的ID看看效果")
    print("   → 运行: python yahboom_pi05_client.py --wrist-camera 2 --exterior-camera 0")
    print("")
    print("3. ⚠️  服务端模型可能只使用了一个摄像头")
    print("   → 检查服务端日志看它是否真的用了两个图片")
    print("")
    print("4. ⚠️  prompt（任务描述）可能不够明确")
    print("   → 尝试更具体的任务描述，比如：")
    print("     - 'pick up the red cube'")
    print("     - 'grasp the object in front of the robot'")
    print("     - 'move the gripper to the object'")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_observation()

