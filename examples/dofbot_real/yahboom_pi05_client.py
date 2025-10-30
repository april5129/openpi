#!/usr/bin/env python3
"""
Linus式极简实现: 连接远程pi0.5模型控制本地机械臂
零废话，零特殊情况，零过度设计
"""

import time
import numpy as np
import cv2
import os
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
    
    def __init__(self, server_host="12.148.158.61", server_port=8000):
        self.arm = Arm_Device()
        time.sleep(0.1)
        
        # 移动到安全位置
        print("🔧 移动机械臂到安全位置...")
        self.arm.Arm_serial_servo_write6(*self.SAFE_POSITION, 1500)  # 慢速移动到安全位置
        time.sleep(2.0)  # 等待移动完成
        print("✅ 机械臂已就位")
        
        # 当前状态 - 使用安全位置初始化
        self.joint_angles = list(self.SAFE_POSITION)
        
        # 并行执行相关状态
        self.joint_angles_lock = threading.Lock()  # 保护关节状态的锁
        self.next_prediction = None  # 存储下一轮预测结果
        self.prediction_lock = threading.Lock()  # 保护预测结果的锁
        self.is_predicting = False  # 是否正在进行预测
        
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
    
    def _save_images(self, original_frame, processed_frame):
        """保存原始图像和处理后的图像"""
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存原始图像（BGR格式）
            original_filename = f"step_{self.step_counter:04d}_{timestamp}_original.jpg"
            original_path = os.path.join(self.images_dir, original_filename)
            cv2.imwrite(original_path, original_frame)
            
            # 保存处理后的图像（需要转换回BGR格式）
            processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            processed_filename = f"step_{self.step_counter:04d}_{timestamp}_processed.jpg"
            processed_path = os.path.join(self.images_dir, processed_filename)
            cv2.imwrite(processed_path, processed_bgr)
            
            print(f"💾 已保存图像: {original_filename} & {processed_filename}")
            
        except Exception as e:
            print(f"⚠️  图像保存失败: {e}")
        
    def normalize_joint_angle(self, joint_idx, angle):
        """将关节角度归一化到[-1, 1]范围"""
        min_angle, max_angle = self.JOINT_LIMITS[joint_idx]
        # 归一化到[-1, 1]: (angle - center) / half_range
        center = (min_angle + max_angle) / 2.0
        half_range = (max_angle - min_angle) / 2.0
        normalized = (angle - center) / half_range
        return max(-1.0, min(1.0, normalized))
    
    def denormalize_joint_angle(self, joint_idx, normalized_value):
        """将归一化值[-1, 1]转换回实际角度"""
        min_angle, max_angle = self.JOINT_LIMITS[joint_idx]
        center = (min_angle + max_angle) / 2.0
        half_range = (max_angle - min_angle) / 2.0
        angle = normalized_value * half_range + center
        return max(min_angle, min(max_angle, angle))

    def get_observation(self, prompt="pick up the object"):
        """获取当前观测 - 图像+关节状态+提示"""
        # 读取摄像头
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # 没摄像头就用黑图
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            original_frame = frame.copy()
        else:
            # 保存原始图像（BGR格式，用于保存）
            original_frame = frame.copy()
            
            # 处理图像用于发送给服务器
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 保存图像到本地
        self._save_images(original_frame, frame)
        
        # 读取关节角度
        for i in range(6):
            angle = self.arm.Arm_serial_servo_read(i + 1)
            if angle is not None:
                self.joint_angles[i] = float(angle)
        
        # 使用改进的归一化算法处理关节位置
        joint_positions = []
        for i in range(5):  # 前5个关节
            if i < len(self.joint_angles):
                normalized = self.normalize_joint_angle(i, self.joint_angles[i])
                joint_positions.append(normalized)
            else:
                joint_positions.append(0.0)
        
        # 补齐到7维 (DROID格式) - 添加两个虚拟腕部关节
        joint_positions.extend([0.0, 0.0])  # 第6、7关节设为0 (无对应硬件)
        
        # 夹爪位置 (单独处理)
        gripper_angle = self.joint_angles[5] if len(self.joint_angles) > 5 else 90.0
        gripper_pos = self.normalize_joint_angle(5, gripper_angle)
        
        # 构建观测 - 按DROID格式
        obs = {
            "observation/exterior_image_1_left": frame,  # numpy数组
            "observation/wrist_image_left": frame,  # 用同一个图像
            "observation/joint_position": np.array(joint_positions, dtype=np.float32),  # 7个关节
            "observation/gripper_position": np.array([gripper_pos], dtype=np.float32),
            "prompt": prompt
        }
        
        # 🔍 详细调试输出 - 打印发送给服务器的所有数据
        print("\n" + "="*60)
        print("📤 发送给服务器的观测数据:")
        print("="*60)
        
        # 打印图像信息
        print(f"🖼️  图像信息:")
        print(f"   - 图像形状: {frame.shape}")
        print(f"   - 图像数据类型: {frame.dtype}")
        
        # 打印原始关节角度
        print(f"🔧 原始关节角度 (度):")
        for i, angle in enumerate(self.joint_angles):
            print(f"   - 关节{i+1}: {angle:.2f}°")
        
        # 打印归一化后的关节位置
        print(f"📐 归一化关节位置 [-1,1]:")
        joint_pos_array = obs["observation/joint_position"]
        for i, pos in enumerate(joint_pos_array):
            if i < 5:
                original_angle = self.joint_angles[i]
                print(f"   - 关节{i+1}: {pos:.3f} (原始: {original_angle:.2f}°)")
            else:
                print(f"   - 关节{i+1}: {pos:.3f} (填充值)")
        
        # 打印夹爪信息
        gripper_array = obs["observation/gripper_position"]
        print(f"🤏 夹爪位置:")
        print(f"   - 原始角度: {self.joint_angles[5]:.2f}°")
        print(f"   - 归一化位置: {gripper_array[0]:.3f}")
        
        # 打印任务提示
        print(f"💬 任务提示: '{prompt}'")
        
        # 打印观测字典的键和数据类型
        print(f"📋 观测数据结构:")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"   - {key}: {type(value).__name__} {value.shape} {value.dtype}")
            else:
                print(f"   - {key}: {type(value).__name__} = '{value}'")
        
        print("="*60)
        
        return obs
    
    def predict_async(self, prompt):
        """异步预测下一轮动作"""
        def prediction_worker():
            try:
                print("🔮 [异步] 开始下一轮预测...")
                observation = self.get_observation(prompt)
                if observation is None:
                    print("⚠️ [异步] 无法获取观测数据")
                    return
                
                start_time = time.time()
                response = self.policy.infer(observation)
                inference_time = time.time() - start_time
                
                with self.prediction_lock:
                    self.next_prediction = {
                        'response': response,
                        'inference_time': inference_time,
                        'step': self.step_counter + 1
                    }
                
                print(f"🔮 [异步] 预测完成，耗时 {inference_time:.3f}s")
                
            except Exception as e:
                print(f"❌ [异步] 预测失败: {e}")
            finally:
                self.is_predicting = False
        
        if not self.is_predicting:
            self.is_predicting = True
            thread = threading.Thread(target=prediction_worker, daemon=True)
            thread.start()
    
    def get_next_prediction(self):
        """获取异步预测的结果"""
        with self.prediction_lock:
            result = self.next_prediction
            self.next_prediction = None
            return result
    
    def execute_action(self, action_data, prompt=None, enable_parallel=True):
        """执行动作序列，支持并行预测下一轮"""
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
        prediction_trigger_step = 7  # 固定在第7步启动预测
        
        print(f"🎯 开始执行动作序列，共 {total_steps} 步")
        if enable_parallel and prompt:
            print(f"📊 并行策略: 第{prediction_trigger_step}步时启动下一轮预测")
        else:
            print("📊 串行执行模式")
        
        # 执行完整的动作序列
        for step_idx, action in enumerate(actions):
            # DROID动作格式: 8维 (7个关节速度 + 1个夹爪位置)
            if len(action) < 8:
                print(f"⚠️ 第{step_idx+1}步动作维度不足: {len(action)}, 期望8个，跳过")
                continue
            
            # 提取关节速度 (前7个) 和夹爪位置 (第8个)
            joint_velocities = action[:7]  # 7个关节的速度
            gripper_position = action[7]   # 夹爪位置
            
            print(f"  🔧 执行第 {step_idx+1}/{len(actions)} 步")
            print(f"    关节速度: {joint_velocities}")
            print(f"    夹爪位置: {gripper_position}")
            
            # 将速度转换为位置增量 (简单积分)
            # 速度范围假设为[-1, 1]，转换为角度增量
            angles = []
            for i in range(5):  # 只处理前5个关节 (对应Dofbot的前5个关节)
                if i < len(joint_velocities):
                    # 当前角度 + 速度增量
                    velocity = joint_velocities[i]
                    # 限制速度增量 (防止过大的跳跃)
                    velocity = max(-0.3, min(0.3, velocity))  # 限制最大速度
                    
                    # 计算新角度 (当前角度 + 速度增量 * 时间步长)
                    current_angle = self.joint_angles[i]
                    angle_increment = velocity * 15.0  # 15度最大增量
                    new_angle = current_angle + angle_increment
                    
                    # 使用精确的关节限制
                    new_angle = self.denormalize_joint_angle(i, 
                        self.normalize_joint_angle(i, new_angle))
                    angles.append(int(new_angle))
                else:
                    angles.append(int(self.joint_angles[i]))
            
            # 处理夹爪 (使用位置控制，不是速度)
            gripper_angle = self.denormalize_joint_angle(5, gripper_position)
            angles.append(int(gripper_angle))
            
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
                600  # 600ms执行时间，平滑但不太慢
            )
            
            # 更新状态
            with self.joint_angles_lock:
                self.joint_angles = [float(a) for a in safe_angles]
            
            # 🔮 关键优化: 在第7步时启动下一轮预测
            if (enable_parallel and prompt and 
                step_idx == prediction_trigger_step - 1 and  # 第7步（索引6）
                not self.is_predicting):                     # 还没开始预测
                print(f"🚀 [并行] 在第{step_idx + 1}步启动下一轮预测")
                self.predict_async(prompt)
            
            # 在第7步时启动异步预测
            if (enable_parallel and prompt and 
                step_idx + 1 == prediction_trigger_step and 
                hasattr(self, 'start_async_prediction')):
                print(f"🔮 第{prediction_trigger_step}步: 启动异步预测下一轮...")
                self.start_async_prediction(prompt)
            
            # 等待动作完成
            time.sleep(0.6)  # 与执行时间匹配
        
        print(f"✅ 动作序列执行完成")
    
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
                
                # 🔮 检查是否有异步预测结果可用
                cached_prediction = self.get_next_prediction()
                
                if cached_prediction:
                    print(f"⚡ 使用异步预测结果 (步骤 {cached_prediction['step']})")
                    action_data = cached_prediction['response']
                    inference_time = cached_prediction['inference_time']
                    
                    # 显示观测数据摘要（仍需要获取当前观测用于显示）
                    obs = self.get_observation(prompt)
                    joint_pos = obs["observation/joint_position"]
                    gripper_pos = obs["observation/gripper_position"]
                    image_shape = obs["observation/exterior_image_1_left"].shape
                    print(f"  📊 观测摘要: 图像{image_shape}, 关节位置{joint_pos.shape}, 夹爪位置{gripper_pos.shape}")
                else:
                    # 没有缓存结果，进行同步预测
                    print("🔄 进行同步预测...")
                    obs = self.get_observation(prompt)
                    
                    # 显示观测数据摘要
                    joint_pos = obs["observation/joint_position"]
                    gripper_pos = obs["observation/gripper_position"]
                    image_shape = obs["observation/exterior_image_1_left"].shape
                    print(f"  📊 观测摘要: 图像{image_shape}, 关节位置{joint_pos.shape}, 夹爪位置{gripper_pos.shape}")
                    
                    # 使用openpi_client发送到服务器并接收动作
                    print("📡 正在发送观测数据到服务器...")
                    start_time = time.time()
                    action_data = self.policy.infer(obs)
                    inference_time = time.time() - start_time
                
                print("📥 收到服务器响应:")
                print(f"   - 响应类型: {type(action_data)}")
                print(f"   - 网络往返时间: {inference_time:.3f}s")
                
                # 显示服务器响应的统计信息
                actions = action_data.get('actions', [])
                print(f"📊 步骤 {self.step_counter} 总结:")
                print(f"   - 动作序列长度: {len(actions)}")
                if len(actions) > 0:
                    print(f"   - 首个动作维度: {len(actions[0])}")
                    print(f"   - 首个动作范围: [{min(actions[0]):.3f}, {max(actions[0]):.3f}]")
                print(f"   - 策略推理时间: {action_data.get('policy_timing', {}).get('infer_ms', 'N/A')} ms")
                print(f"   - 服务器推理时间: {action_data.get('server_timing', {}).get('infer_ms', 'N/A')} ms")
                
                # 执行动作序列
                self.execute_action(action_data)
                
                # 显示执行后的关节状态
                self.print_joint_status()
                
                # 不需要额外的sleep，因为execute_action已经包含了等待时间
                    
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
    parser.add_argument("--host", default="12.148.158.61", help="服务器IP")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--task", default="pick up the object", help="任务描述")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    
    args = parser.parse_args()
    
    print("🤖 Yahboom机械臂 + pi0.5远程控制系统")
    print(f"服务器: {args.host}:{args.port}")
    print(f"任务: {args.task}")
    
    client = YahboomPi05Client(args.host, args.port)
    client.run(args.task)


if __name__ == "__main__":
    main()