#!/usr/bin/env python3
"""
测试摄像头配置 - 帮助确定哪个摄像头对应哪个ID
"""

import cv2
import numpy as np
import os

def test_camera(camera_id, save_dir="examples/dofbot_real/images"):
    """测试指定ID的摄像头"""
    print(f"\n测试摄像头 ID={camera_id}")
    print("-" * 50)
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return False
    
    # 读取一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 摄像头 {camera_id} 无法读取图像")
        return False
    
    # 显示摄像头信息
    print(f"✅ 摄像头 {camera_id} 工作正常")
    print(f"   - 分辨率: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   - 颜色通道: {frame.shape[2]}")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存测试图像到 images 文件夹
    filename = f"test_camera_{camera_id}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"   - 已保存测试图像: {filepath}")
    print(f"   - 请查看图像确认这是哪个摄像头")
    
    return True

def main():
    print("="*50)
    print("摄像头配置测试工具")
    print("="*50)
    
    print("\n正在测试系统中的摄像头...")
    print("  - 测试摄像头 ID 0 和 2")
    
    # 只测试已知的两个摄像头
    cam0_ok = test_camera(0)  # Microdia USB 2.0 Camera (机械臂上)
    cam2_ok = test_camera(2)  # Realtek Integrated Webcam (空中全局)
    
    print("\n" + "="*50)
    print("测试总结:")
    print("="*50)
    print(f"摄像头 0 (机械臂摄像头): {'✅ 正常' if cam0_ok else '❌ 失败'}")
    print(f"摄像头 2 (全局摄像头): {'✅ 正常' if cam2_ok else '❌ 失败'}")
    
    print("\n📝 配置说明:")
    if cam0_ok and cam2_ok:
        print("✅ 两个摄像头都正常工作！")
        print("\n当前配置:")
        print("   - 机械臂摄像头 (wrist): ID 0 - Microdia USB 2.0 Camera")
        print("   - 全局摄像头 (exterior): ID 2 - Realtek Integrated Webcam")
        print("\n📸 请查看 images/test_camera_*.jpg 确认摄像头视角")
        print("\n▶️  可以直接运行主程序:")
        print("   python yahboom_pi05_client.py")
    elif cam0_ok:
        print("⚠️ 只检测到机械臂摄像头（ID 0），全局摄像头（ID 2）失败")
    elif cam2_ok:
        print("⚠️ 只检测到全局摄像头（ID 2），机械臂摄像头（ID 0）失败")
    else:
        print("❌ 两个摄像头都无法访问，请检查硬件连接")
    
    print(f"\n💾 测试图像已保存到 images/ 文件夹")

if __name__ == "__main__":
    main()

