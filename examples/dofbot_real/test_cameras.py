#!/usr/bin/env python3
"""
测试摄像头配置 - 帮助确定哪个摄像头对应哪个ID
"""

import cv2
import numpy as np

def test_camera(camera_id):
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
    
    # 保存测试图像
    filename = f"test_camera_{camera_id}.jpg"
    cv2.imwrite(filename, frame)
    print(f"   - 已保存测试图像: {filename}")
    print(f"   - 请查看图像确认这是哪个摄像头")
    
    return True

def main():
    print("="*50)
    print("摄像头配置测试工具")
    print("="*50)
    
    print("\n根据你的描述:")
    print("  - 摄像头 0 应该是: Microdia USB 2.0 Camera (机械臂上)")
    print("  - 摄像头 1 应该是: Realtek Integrated Webcam (空中全局)")
    
    # 测试摄像头 0
    cam0_ok = test_camera(0)
    
    # 测试摄像头 1
    cam1_ok = test_camera(1)
    
    # 尝试测试摄像头 2 (以防万一)
    cam2_ok = test_camera(2)
    
    print("\n" + "="*50)
    print("测试总结:")
    print("="*50)
    print(f"摄像头 0: {'✅ 正常' if cam0_ok else '❌ 失败'}")
    print(f"摄像头 1: {'✅ 正常' if cam1_ok else '❌ 失败'}")
    print(f"摄像头 2: {'✅ 正常' if cam2_ok else '❌ 失败'}")
    
    print("\n📝 下一步:")
    print("1. 查看生成的 test_camera_*.jpg 图像")
    print("2. 确定哪个ID对应哪个摄像头")
    print("3. 如果ID不对，运行主程序时使用以下参数:")
    print("   python yahboom_pi05_client.py --wrist-camera <ID> --exterior-camera <ID>")
    print("\n例如，如果机械臂摄像头是ID 1，全局摄像头是ID 0:")
    print("   python yahboom_pi05_client.py --wrist-camera 1 --exterior-camera 0")

if __name__ == "__main__":
    main()

