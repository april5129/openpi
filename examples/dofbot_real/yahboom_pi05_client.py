#!/usr/bin/env python3
"""
Linus式极简实现: 连接远程pi0.5模型控制本地机械臂
零废话，零特殊情况，零过度设计
"""

import time
import numpy as np
import cv2
import os
import json
import threading
from datetime import datetime
from Arm_Lib import Arm_Device
from openpi_client import websocket_client_policy


class YahboomPi05Client:
    """最简WebSocket客户端 - 一个类解决所有问题"""
    
    # Dofbot关节定义和限制 (基于Yahboom Dofbot 6DOF规格)
    JOINT_NAMES = [
        "base_rotation",      # 关节1: 底座旋转 (0-180°)
        "shoulder",           # 关节2: 肩部 (0-180°) 
        "elbow",             # 关节3: 肘部 (0-180°)
        "wrist_pitch",       # 关节4: 腕部俯仰 (0-180°)
        "wrist_roll",        # 关节5: 腕部翻滚 (0-180°)
        "gripper"            # 关节6: 夹爪 (0-180°)
    ]
    
    # 每个关节的实际角度范围 [min, max] (度)
    JOINT_LIMITS = [
        [0, 180],    # base_rotation: 底座可全范围旋转
        [0, 180],    # shoulder: 肩部关节范围
        [0, 180],    # elbow: 肘部关节范围  
        [0, 180],    # wrist_pitch: 腕部俯仰范围
        [0, 180],    # wrist_roll: 腕部翻滚范围
        [0, 180]     # gripper: 夹爪开合范围
    ]
    
    # 安全的初始位置 (度) - 避免奇异点和碰撞
    SAFE_POSITION = [90, 135, 0, 1, 89, 3]  # 更安全的姿态
    
    def __init__(self, server_host="wss://torie-nonefficient-darkly.ngrok-free.dev", server_port=443, 
                 wrist_camera_id=0, exterior_camera_id=2, action_horizon=30):
        self.arm = Arm_Device()
        time.sleep(0.1)
        
        # 加载归一化统计数据
        norm_stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
        with open(norm_stats_path, 'r') as f:
            norm_data = json.load(f)
            self.state_stats = norm_data['norm_stats']['state']
            self.action_stats = norm_data['norm_stats']['actions']
        print(f"📊 已加载归一化统计数据: {norm_stats_path}")
        
        # 摄像头配置
        self.wrist_camera_id = wrist_camera_id      # 机械臂上的摄像头 (Microdia USB 2.0 Camera)
        self.exterior_camera_id = exterior_camera_id  # 空中全局摄像头 (Realtek Integrated Webcam)
        self.action_horizon = action_horizon  # 每次预测的动作步数
        print(f"📷 摄像头配置:")
        print(f"   - 机械臂摄像头 (wrist): /dev/video{wrist_camera_id}")
        print(f"   - 全局摄像头 (exterior): /dev/video{exterior_camera_id}")
        print(f"🎯 动作预测步数: {action_horizon} 步")
        
        # 移动到安全位置
        print("🔧 移动机械臂到安全位置...")
        self.arm.Arm_serial_servo_write6(*self.SAFE_POSITION, 1500)  # 慢速移动到安全位置
        time.sleep(2.0)  # 等待移动完成
        print("✅ 机械臂已就位")
        
        # 当前状态 - 使用安全位置初始化
        self.joint_angles = list(self.SAFE_POSITION)
        
        # 状态管理
        self.joint_angles_lock = threading.Lock()  # 保护关节状态的锁
        
        # 设置图像保存目录
        self.images_dir = "/home/yahboom/openpi/examples/dofbot_real/images"
        os.makedirs(self.images_dir, exist_ok=True)
        self.step_counter = 0  # 用于图像文件命名
        print(f"📁 图像保存目录: {self.images_dir}")
        
        # 连接到远程pi0.5服务器
        print(f"连接到远程服务器: {server_host}:{server_port}")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_host,
            port=server_port
        )
        print(f"✅ 连接成功! 服务器元数据: {self.policy.get_server_metadata()}")
    
    def _save_images(self, wrist_original, wrist_processed, exterior_original, exterior_processed):
        """保存两个摄像头的原始图像和处理后的图像"""
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存机械臂摄像头图像（BGR格式）
            wrist_original_filename = f"step_{self.step_counter:04d}_{timestamp}_wrist_original.jpg"
            wrist_original_path = os.path.join(self.images_dir, wrist_original_filename)
            cv2.imwrite(wrist_original_path, wrist_original)
            
            # 保存机械臂摄像头处理后的图像（需要转换回BGR格式）
            wrist_processed_bgr = cv2.cvtColor(wrist_processed, cv2.COLOR_RGB2BGR)
            wrist_processed_filename = f"step_{self.step_counter:04d}_{timestamp}_wrist_processed.jpg"
            wrist_processed_path = os.path.join(self.images_dir, wrist_processed_filename)
            cv2.imwrite(wrist_processed_path, wrist_processed_bgr)
            
            # 保存全局摄像头图像（BGR格式）
            exterior_original_filename = f"step_{self.step_counter:04d}_{timestamp}_exterior_original.jpg"
            exterior_original_path = os.path.join(self.images_dir, exterior_original_filename)
            cv2.imwrite(exterior_original_path, exterior_original)
            
            # 保存全局摄像头处理后的图像（需要转换回BGR格式）
            exterior_processed_bgr = cv2.cvtColor(exterior_processed, cv2.COLOR_RGB2BGR)
            exterior_processed_filename = f"step_{self.step_counter:04d}_{timestamp}_exterior_processed.jpg"
            exterior_processed_path = os.path.join(self.images_dir, exterior_processed_filename)
            cv2.imwrite(exterior_processed_path, exterior_processed_bgr)
            
            print(f"💾 已保存图像:")
            print(f"   - 机械臂视角: {wrist_original_filename} & {wrist_processed_filename}")
            print(f"   - 全局视角: {exterior_original_filename} & {exterior_processed_filename}")
            
        except Exception as e:
            print(f"⚠️  图像保存失败: {e}")
        
    def normalize_state(self, state_vector):
        """使用 z-score 归一化状态（关节位置和夹爪位置）
        state_vector: [joint_pos_0, ..., joint_pos_6, gripper_pos]
        使用 state_stats 进行归一化
        """
        mean = np.array(self.state_stats['mean'])
        std = np.array(self.state_stats['std'])
        normalized = (state_vector - mean) / (std + 1e-6)
        return normalized
    
    def denormalize_action(self, action_vector):
        """反归一化动作（从服务器返回的归一化动作转换为实际动作）
        action_vector: [joint_vel_0, ..., joint_vel_6, gripper_pos]
        使用 action_stats 进行反归一化
        """
        mean = np.array(self.action_stats['mean'])
        std = np.array(self.action_stats['std'])
        denormalized = action_vector * (std + 1e-6) + mean
        return denormalized

    def get_observation(self, prompt="pick up the object"):
        """获取当前观测 - 图像+关节状态+提示"""
        # 读取机械臂摄像头 (wrist camera)
        wrist_cap = cv2.VideoCapture(self.wrist_camera_id)
        wrist_ret, wrist_frame = wrist_cap.read()
        wrist_cap.release()
        
        # 读取全局摄像头 (exterior camera)
        exterior_cap = cv2.VideoCapture(self.exterior_camera_id)
        exterior_ret, exterior_frame = exterior_cap.read()
        exterior_cap.release()
        
        # 处理机械臂摄像头图像
        if not wrist_ret:
            print("⚠️ 机械臂摄像头读取失败，使用黑色图像")
            wrist_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            wrist_original = wrist_frame.copy()
        else:
            wrist_original = wrist_frame.copy()
            wrist_frame = cv2.resize(wrist_frame, (224, 224))
            wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
        
        # 处理全局摄像头图像
        if not exterior_ret:
            print("⚠️ 全局摄像头读取失败，使用黑色图像")
            exterior_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            exterior_original = exterior_frame.copy()
        else:
            exterior_original = exterior_frame.copy()
            exterior_frame = cv2.resize(exterior_frame, (224, 224))
            exterior_frame = cv2.cvtColor(exterior_frame, cv2.COLOR_BGR2RGB)
        
        # 保存图像到本地
        self._save_images(wrist_original, wrist_frame, exterior_original, exterior_frame)
        
        # 读取关节角度
        for i in range(6):
            angle = self.arm.Arm_serial_servo_read(i + 1)
            if angle is not None:
                self.joint_angles[i] = float(angle)
        
        # 构建原始状态向量 (DROID格式: 7个关节位置 + 1个夹爪位置)
        # Dofbot只有5个关节 + 1个夹爪，需要补齐到7个关节
        raw_state = np.zeros(8, dtype=np.float32)
        
        # 填充前5个关节（Dofbot的实际关节）
        for i in range(5):
            if i < len(self.joint_angles):
                raw_state[i] = self.joint_angles[i]
        
        # 第6、7个关节设为0（Dofbot没有这些关节）
        raw_state[5] = 0.0
        raw_state[6] = 0.0
        
        # 第8维：夹爪位置
        gripper_angle = self.joint_angles[5] if len(self.joint_angles) > 5 else 90.0
        raw_state[7] = gripper_angle
        
        # 使用 norm_stats 进行归一化
        normalized_state = self.normalize_state(raw_state)
        
        # 分离成 joint_position 和 gripper_position
        joint_positions = normalized_state[:7]  # 前7维
        gripper_pos = normalized_state[7:8]     # 第8维
        
        # 构建观测 - 按DROID格式
        obs = {
            "observation/exterior_image_1_left": exterior_frame,  # 全局摄像头图像
            "observation/wrist_image_left": wrist_frame,  # 机械臂摄像头图像
            "observation/joint_position": joint_positions.astype(np.float32),  # 7个关节
            "observation/gripper_position": gripper_pos.astype(np.float32),    # 1个夹爪
            "prompt": prompt,
            "action_horizon": self.action_horizon  # 指定预测步数
        }
        
        # 🔍 详细调试输出 - 打印发送给服务器的所有数据
        print("\n" + "="*60)
        print("📤 发送给服务器的观测数据:")
        print("="*60)
        
        # 打印图像信息
        print(f"🖼️  图像信息:")
        print(f"   - 机械臂摄像头 (wrist): {wrist_frame.shape}, {wrist_frame.dtype}")
        print(f"   - 全局摄像头 (exterior): {exterior_frame.shape}, {exterior_frame.dtype}")
        
        # 打印原始关节角度
        print(f"🔧 原始关节角度 (度):")
        for i, angle in enumerate(self.joint_angles):
            print(f"   - 关节{i+1}: {angle:.2f}°")
        
        # 打印归一化后的关节位置
        print(f"📐 归一化状态向量 (使用 norm_stats):")
        print(f"   - 原始状态: {raw_state}")
        print(f"   - 归一化后: {normalized_state}")
        print(f"   - joint_position (前7维): {joint_positions}")
        print(f"   - gripper_position (第8维): {gripper_pos}")
        
        # 打印任务提示
        print(f"💬 任务提示: '{prompt}'")
        print(f"🎯 动作预测步数: {self.action_horizon} 步")
        
        # 打印观测字典的键和数据类型
        print(f"📋 观测数据结构:")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"   - {key}: {type(value).__name__} {value.shape} {value.dtype}")
            else:
                print(f"   - {key}: {type(value).__name__} = {value}")
        
        print("="*60)
        
        return obs
    
    def execute_action(self, action_data, steps_to_execute=15):
        """执行动作序列的前N步"""
        if action_data is None:
            print("⚠️ 收到空的动作数据")
            return
            
        if not isinstance(action_data, dict) or "actions" not in action_data:
            print(f"⚠️ 动作数据中没有 'actions' 字段，可用字段: {list(action_data.keys())}")
            return
            
        actions = action_data["actions"]
        
        # 处理动作格式
        if len(actions) == 0:
            print("⚠️ 收到空动作")
            return
        
        total_steps = len(actions)
        steps_to_execute = min(steps_to_execute, total_steps)
        
        print(f"🎯 收到 {total_steps} 步动作序列，执行前 {steps_to_execute} 步")
        
        # 只执行前N步
        for step_idx in range(steps_to_execute):
            action = np.array(actions[step_idx])
            # DROID动作格式: 8维 (7个关节速度 + 1个夹爪位置)
            if len(action) < 8:
                print(f"⚠️ 第{step_idx+1}步动作维度不足: {len(action)}, 期望8个，跳过")
                continue
            
            # 反归一化动作：从服务器返回的归一化动作 → 实际动作
            denorm_action = self.denormalize_action(action)
            
            # 提取关节速度 (前7个) 和夹爪位置 (第8个)
            joint_velocities = denorm_action[:7]  # 7个关节的速度（已反归一化）
            gripper_position = denorm_action[7]   # 夹爪位置（已反归一化）
            
            print(f"  🔧 执行第 {step_idx + 1}/{steps_to_execute} 步:")
            print(f"    归一化动作: {action}")
            print(f"    反归一化后:")
            print(f"      关节速度: {joint_velocities}")
            print(f"      夹爪位置: {gripper_position}")
            
            # 将速度转换为位置增量 (简单积分)
            # DROID 控制频率: 30Hz，时间步长 dt = 1/30 秒
            dt = 1.0 / 30.0  # 时间步长（秒）
            
            angles = []
            for i in range(5):  # 只处理前5个关节 (对应Dofbot的前5个关节)
                if i < len(joint_velocities):
                    # 当前角度 + 速度增量
                    velocity = joint_velocities[i]
                    
                    # 计算新角度: 当前角度 + 速度 * 时间步长
                    current_angle = self.joint_angles[i]
                    angle_increment = velocity * dt
                    new_angle = current_angle + angle_increment
                    
                    # 限制角度在有效范围内
                    min_angle, max_angle = self.JOINT_LIMITS[i]
                    new_angle = max(min_angle, min(max_angle, new_angle))
                    angles.append(int(new_angle))
                else:
                    angles.append(int(self.joint_angles[i]))
            
            # 处理夹爪 (gripper_position 是目标夹爪角度)
            gripper_angle = max(0, min(180, int(gripper_position)))
            angles.append(gripper_angle)
            
            # 安全检查 - 确保所有角度在有效范围内
            safe_angles = []
            for i, angle in enumerate(angles):
                min_angle, max_angle = self.JOINT_LIMITS[i]
                safe_angle = max(min_angle, min(max_angle, angle))
                safe_angles.append(safe_angle)
            
            print(f"    目标角度: {safe_angles}")
            
            # 执行动作
            self.arm.Arm_serial_servo_write6(
                safe_angles[0], safe_angles[1], safe_angles[2], 
                safe_angles[3], safe_angles[4], safe_angles[5], 
                600  # 600ms执行时间
            )
            
            # 更新状态
            with self.joint_angles_lock:
                self.joint_angles = [float(a) for a in safe_angles]
            
            # 等待动作完成
            time.sleep(0.6)  # 与执行时间匹配
        
        print(f"✅ 执行完成 {steps_to_execute} 步动作")
    
    def print_joint_status(self):
        """打印当前关节状态"""
        print("🔧 当前关节状态:")
        for i, (name, angle) in enumerate(zip(self.JOINT_NAMES, self.joint_angles)):
            min_angle, max_angle = self.JOINT_LIMITS[i]
            normalized = self.normalize_joint_angle(i, angle)
            print(f"  {name:15}: {angle:6.1f}° (范围: {min_angle:3.0f}-{max_angle:3.0f}°, 归一化: {normalized:+.3f})")

    def run(self, prompt="pick up the red block"):
        """主循环 - 使用openpi_client执行控制"""
        print(f"🚀 开始执行任务: '{prompt}'")
        print(f"🤖 Dofbot配置: {len(self.JOINT_NAMES)}个关节")
        self.print_joint_status()
        
        try:
            while True:
                self.step_counter += 1
                print(f"\n🚀 === 步骤 {self.step_counter} 开始 ===")
                
                start_time = time.time()
                
                # 1️⃣ 获取当前观测（图像 + 关节状态）
                print("📸 采集观测数据...")
                obs = self.get_observation(prompt)
                
                # 2️⃣ 发送到服务器预测30步动作
                print("📡 正在发送观测数据到服务器...")
                inference_start = time.time()
                action_data = self.policy.infer(obs)
                inference_time = time.time() - inference_start
                
                # 3️⃣ 显示服务器响应
                actions = action_data.get('actions', [])
                print(f"📥 收到动作预测: 共 {len(actions)} 步 (推理耗时: {inference_time:.3f}s)")
                
                # 4️⃣ 执行前15步动作
                self.execute_action(action_data, steps_to_execute=15)
                
                # 5️⃣ 显示执行后的关节状态
                self.print_joint_status()
                
                print(f"⏱️  本轮总耗时: {time.time() - start_time:.3f}s")
                
                    
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        except Exception as e:
            print(f"❌ 错误: {e}")
        finally:
            # 回到安全位置
            print("🔧 返回安全位置...")
            self.arm.Arm_serial_servo_write6(*self.SAFE_POSITION, 1000)
            time.sleep(1.0)
            del self.arm
            print("🧹 清理完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Yahboom机械臂 + pi0.5远程控制")
    parser.add_argument("--host", default="wss://torie-nonefficient-darkly.ngrok-free.dev", help="服务器IP")
    parser.add_argument("--port", type=int, default=443, help="服务器端口")
    parser.add_argument("--prompt", default="pick up the object", help="任务描述")
    parser.add_argument("--wrist-camera", type=int, default=0, help="机械臂摄像头ID (Microdia USB 2.0 Camera)")
    parser.add_argument("--exterior-camera", type=int, default=2, help="全局摄像头ID (Realtek Integrated Webcam)")
    parser.add_argument("--action-horizon", type=int, default=30, help="每次预测的动作步数")
    
    args = parser.parse_args()
    
    print("🤖 Yahboom机械臂 + pi0.5远程控制系统")
    print(f"服务器: {args.host}:{args.port}")
    print(f"任务: {args.prompt}")
    
    client = YahboomPi05Client(
        args.host, 
        args.port, 
        wrist_camera_id=args.wrist_camera,
        exterior_camera_id=args.exterior_camera,
        action_horizon=args.action_horizon
    )
    client.run(args.prompt)


if __name__ == "__main__":
    main()
