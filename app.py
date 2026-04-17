# REAL-TIME HYBRID ASL RECOGNITION
# Bật môi trường ảo: .\asl_env\Scripts\activate
# Chạy: python app.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
import time
import json

class HybridASLRecognizer:
    def __init__(self, cnn_model_path, bilstm_model_path, metadata_path, 
                 img_size=224, seq_length=30):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.seq_length = seq_length
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.class_mapping = self.metadata['class_mapping']
        self.idx_to_class = {int(k): v for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Load CNN model (cho static)
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        
        # Load BiLSTM model (cho dynamic)
        self.bilstm_model = self._load_bilstm_model(bilstm_model_path)
        
        # Keypoints buffer cho BiLSTM
        self.keypoints_buffer = deque(maxlen=seq_length)
        
        # Motion detection
        self.prev_keypoints = None
        self.motion_threshold = 0.05
        
        # State
        self.current_mode = "static"  # static hoặc dynamic
        self.static_counter = 0
        self.dynamic_counter = 0
        self.force_mode = "auto"  # auto, static, hoặc dynamic
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Colors for fingers
        self.finger_colors = {
            'thumb': (0, 255, 255), 'index': (255, 0, 255), 'middle': (0, 255, 0),
            'ring': (255, 0, 0), 'pinky': (0, 165, 255), 'palm': (255, 255, 255)
        }
        
        # Hand connections
        self.hand_connections = [
            (0, 1, 'palm'), (0, 5, 'palm'), (0, 9, 'palm'), (0, 13, 'palm'), (0, 17, 'palm'),
            (5, 9, 'palm'), (9, 13, 'palm'), (13, 17, 'palm'),
            (1, 2, 'thumb'), (2, 3, 'thumb'), (3, 4, 'thumb'),
            (5, 6, 'index'), (6, 7, 'index'), (7, 8, 'index'),
            (9, 10, 'middle'), (10, 11, 'middle'), (11, 12, 'middle'),
            (13, 14, 'ring'), (14, 15, 'ring'), (15, 16, 'ring'),
            (17, 18, 'pinky'), (18, 19, 'pinky'), (19, 20, 'pinky')
        ]
        
        # Smoothing - buffer lớn hơn giúp ổn định hơn
        self.prediction_buffer = deque(maxlen=10)
        
        print(f"Hybrid recognizer initialized")
        print(f"Device: {self.device}")
        print(f"Mode: Static (CNN) + Dynamic (BiLSTM)")
    
    def _load_cnn_model(self, model_path):
        """Load CNN model cho static recognition"""
        class LightweightASL_CNN(nn.Module):
            def __init__(self, num_classes=29, img_size=224):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                    nn.Dropout(0.2), nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)
        
        model = LightweightASL_CNN(num_classes=self.num_classes, img_size=self.img_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_bilstm_model(self, model_path):
        """Load BiLSTM model cho dynamic recognition"""
        class AdditiveAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.Linear(hidden_size, 1)
            def forward(self, x):
                weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
                return torch.sum(x * weights.unsqueeze(-1), dim=1), weights
                
        class MultiHeadAttention(nn.Module):
            def __init__(self, hidden_size, num_heads=4):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)
                self.out = nn.Linear(hidden_size, hidden_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attention_weights = torch.softmax(scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                context = torch.matmul(attention_weights, V)
                context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
                return self.out(context), attention_weights

        class BiLSTM_Attention(nn.Module):
            def __init__(self, input_size=75, hidden_size=128, num_layers=2, num_classes=29, dropout=0.3, attention_type='multihead'):
                super().__init__()
                self.layer_norm_in = nn.LayerNorm(input_size)
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
                if attention_type == 'additive':
                    self.attention = AdditiveAttention(hidden_size * 2)
                else:
                    self.attention = MultiHeadAttention(hidden_size * 2, num_heads=4)
                
                self.fc1 = nn.Linear(hidden_size * 2, 128)
                self.layer_norm_out = nn.LayerNorm(128)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = self.layer_norm_in(x)
                lstm_out, _ = self.lstm(x)
                context, attention_weights = self.attention(lstm_out)
                
                # Global Average Pooling for MultiHeadAttention output if 3D
                if context.dim() == 3:
                    context = context.mean(dim=1)
                
                out = self.fc1(context)
                out = self.layer_norm_out(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return out
        
        model = BiLSTM_Attention(input_size=75, num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model
    
    def create_bone_diagram(self, hand_landmarks):
        """Tạo bone diagram từ hand landmarks (với Auto-Scaling chống nhiễu xa / gần)"""
        # Tính Box bao quanh bàn tay để tự động Zoom
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Thêm 20% biên giới (padding)
        size = max(max_x - min_x, max_y - min_y) * 1.5
        if size == 0: size = 1.0 # Tránh lỗi chia 0
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        h, w = self.img_size, self.img_size
        
        points = []
        for x, y in zip(x_coords, y_coords):
            # Scale đưa bàn tay luôn nằm mượt mà giữa khung ảnh 224x224
            norm_x = (x - center_x) / size + 0.5
            norm_y = (y - center_y) / size + 0.5
            points.append((int(norm_x * w), int(norm_y * h)))
        
        for connection in self.hand_connections:
            pt1_idx, pt2_idx, part = connection
            if pt1_idx < len(points) and pt2_idx < len(points):
                color = self.finger_colors[part]
                cv2.line(img, points[pt1_idx], points[pt2_idx], color, 1, cv2.LINE_AA)
        
        for pt in points:
            cv2.circle(img, pt, 2, (255, 255, 255), -1)
        
        return img
    
    def extract_keypoints(self, hand_landmarks):
        """Trích xuất keypoints ROBUST (Wrist-relative + Distances + Crossing)"""
        # 1. Cơ bản: Wrist-relative landmarks (63 features)
        wrist = hand_landmarks.landmark[0]
        kp_list = []
        for lm in hand_landmarks.landmark:
            kp_list.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        
        kp_array = np.array(kp_list) # (21, 3)
        features = kp_array.flatten() # (63,)
        
        # 2. Đặc trưng khoảng cách đầu ngón (10 features)
        # Các chỉ số đầu ngón: 4(thumb), 8(index), 12(middle), 16(ring), 20(pinky)
        tips = [4, 8, 12, 16, 20]
        extra_dists = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                p1 = kp_array[tips[i]]
                p2 = kp_array[tips[j]]
                dist = np.linalg.norm(p1 - p2)
                extra_dists.append(dist)
        
        # 3. Đặc trưng bắt chéo (Phân biệt U và R) (2 features)
        # Dùng np.sign để đồng nhất hoàn toàn với logic trong notebook
        diff_x = kp_array[12][0] - kp_array[8][0]
        is_crossed_x = np.sign(diff_x)
        
        diff_z = kp_array[12][2] - kp_array[8][2]
        is_crossed_z = np.sign(diff_z)
        
        # Gộp tất cả: 63 + 10 + 2 = 75 features
        final_features = np.concatenate([
            features, 
            np.array(extra_dists), 
            np.array([is_crossed_x, is_crossed_z])
        ])
        
        return final_features
    
    def detect_motion(self, current_keypoints):
        """Phát hiện chuyển động để quyết định dùng model nào"""
        if self.prev_keypoints is None:
            self.prev_keypoints = current_keypoints
            return False
        
        # Tính movement
        movement = np.mean(np.abs(current_keypoints - self.prev_keypoints))
        self.prev_keypoints = current_keypoints.copy()
        
        return movement > self.motion_threshold
    
    def predict_static(self, bone_img):
        """Dự đoán bằng CNN với TTA + Temperature Scaling"""
        img = cv2.resize(bone_img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize: mean=0.5, std=0.5
        
        # TTA: Dự đoán cả ảnh gốc và ảnh lật ngang, lấy trung bình logits
        img_orig = np.transpose(img, (2, 0, 1))
        img_flip = np.transpose(img[:, ::-1, :].copy(), (2, 0, 1))  # Flip ngang
        
        batch = torch.FloatTensor(np.stack([img_orig, img_flip])).to(self.device)
        
        with torch.no_grad():
            logits = self.cnn_model(batch)
            avg_logits = logits.mean(dim=0)  # Trung bình logits (trước softmax)
            
            # Temperature Scaling: T < 1 làm nhọn phân phối, tăng confidence
            # T=0.35 phù hợp khi confidence thực tế 20-40% mà nhận đúng
            TEMPERATURE = 0.35
            scaled_logits = avg_logits / TEMPERATURE
            probs = torch.softmax(scaled_logits, dim=0)
            confidence, predicted = torch.max(probs, 0)
        
        return self.idx_to_class[predicted.item()], confidence.item() * 100
    
    def predict_dynamic(self):
        """Dự đoán bằng BiLSTM (cho ký tự động J, Z)"""
        if len(self.keypoints_buffer) < self.seq_length:
            return f"Loading_Buffer_{len(self.keypoints_buffer)}", 0.0
        
        # Tạo sequence từ buffer
        sequence = np.array(self.keypoints_buffer)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.bilstm_model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_char = self.idx_to_class[predicted.item()]
        
        # Trả về tất cả các ký tự (không giới hạn chỉ J và Z nữa)
        return predicted_char, confidence.item() * 100
    
    def process_frame(self, frame):
        """Xử lý một frame từ webcam"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Vẽ landmarks lên frame
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Tạo bone diagram
            bone_img = self.create_bone_diagram(hand_landmarks)
            
            # Trích xuất keypoints
            keypoints = self.extract_keypoints(hand_landmarks)
            self.keypoints_buffer.append(keypoints)
            
            # Phát hiện chuyển động (hoặc ép chế độ bằng tay)
            if self.force_mode == "static":
                is_moving = False
            elif self.force_mode == "dynamic":
                is_moving = True
            else:
                is_moving = self.detect_motion(keypoints)
            
            # Quyết định dùng model nào
            prediction = None
            confidence = 0.0
            
            prev_mode = self.current_mode
            
            if is_moving:
                # Đang có chuyển động -> dùng BiLSTM
                self.dynamic_counter += 1
                self.static_counter = 0
                
                if self.dynamic_counter > 5:  # Giảm độ trễ
                    raw_pred, confidence = self.predict_dynamic()
                    # Lọc bỏ chuỗi Loading_Buffer (buffer chưa đủ frames)
                    if not raw_pred.startswith("Loading_Buffer"):
                        prediction = raw_pred
                        self.current_mode = "dynamic"
            else:
                # Tĩnh -> dùng CNN
                self.static_counter += 1
                self.dynamic_counter = 0
                
                if self.static_counter > 2:  # Giảm độ trễ
                    prediction, confidence = self.predict_static(bone_img)
                    self.current_mode = "static"
            
            # Clear buffer khi đổi mode để tránh lẫn lộn CNN/BiLSTM
            if self.current_mode != prev_mode:
                self.prediction_buffer.clear()
            
            # Lưu vào buffer để làm mượt (tránh nháy chữ)
            if prediction is not None:
                self.prediction_buffer.append((prediction, confidence))
                
            # Smoothing: vote có trọng số theo confidence
            if len(self.prediction_buffer) > 0:
                # Tính tổng confidence cho từng class (weighted voting)
                score_map = {}
                for pred, conf in self.prediction_buffer:
                    score_map[pred] = score_map.get(pred, 0) + conf
                
                # Chọn class có tổng confidence cao nhất
                best_pred = max(score_map, key=score_map.get)
                confs = [c for p, c in self.prediction_buffer if p == best_pred]
                
                prediction = best_pred
                confidence = sum(confs) / len(confs)
            
            return frame, bone_img, prediction, confidence, self.current_mode
        
        # Nếu không có tay: Xóa bộ nhớ đệm
        self.prediction_buffer.clear()
        return frame, None, None, 0.0, "no_hand"
    
    def release(self):
        self.hands.close()

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    # Đường dẫn đến model files (tải từ Kaggle)
    CNN_MODEL_PATH = "models/best_cnn_model.pth"  # Từ Notebook 3
    BILSTM_MODEL_PATH = "models/best_bilstm_model.pth"  # Từ Notebook 4
    METADATA_PATH = "models/bone_metadata.json"  # Từ Notebook 2
    
    # Khởi tạo recognizer
    recognizer = HybridASLRecognizer(
        cnn_model_path=CNN_MODEL_PATH,
        bilstm_model_path=BILSTM_MODEL_PATH,
        metadata_path=METADATA_PATH,
        img_size=224,
        seq_length=30
    )
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("\n" + "="*50)
    print("HYBRID ASL RECOGNITION - REAL TIME")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("  'm' - Switch to manual mode (CNN/BiLSTM)")
    print("="*50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame, bone_img, prediction, confidence, mode = recognizer.process_frame(frame)
        
        # FPS calculation
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = end_time
        
        # Display info
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode indicator
        mode_colors = {"static": (255, 255, 0), "dynamic": (0, 255, 255), "no_hand": (128, 128, 128)}
        mode_str = f"Mode: {mode.upper()}"
        if recognizer.force_mode != "auto":
            mode_str += f" (MANUAL)"
            
        cv2.putText(processed_frame, mode_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_colors.get(mode, (128, 128, 128)), 2)
        
        # Prediction
        if prediction is not None:
            color = (0, 255, 0) if confidence > 80 else ((0, 255, 255) if confidence > 60 else (0, 0, 255))
            text = f"Prediction: {prediction} ({confidence:.1f}%)"
            cv2.putText(processed_frame, text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Confidence bar
            bar_width = int(confidence * 2)
            cv2.rectangle(processed_frame, (10, 120), (10 + bar_width, 140), color, -1)
            cv2.rectangle(processed_frame, (10, 120), (210, 140), (255, 255, 255), 1)
        
        # Show bone diagram
        if bone_img is not None:
            bone_img_resized = cv2.resize(bone_img, (224, 224))
            cv2.imshow('Bone Diagram', bone_img_resized)
        
        # Show main frame
        cv2.imshow('Hybrid ASL Recognition', processed_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and bone_img is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.png", processed_frame)
            print(f"Screenshot saved")
        elif key == ord('m'):
            if recognizer.force_mode == "auto":
                recognizer.force_mode = "static"
                print("\n--> MANUAL OVERRIDE: ép cứng chạy STATIC (CNN)")
            elif recognizer.force_mode == "static":
                recognizer.force_mode = "dynamic"
                print("\n--> MANUAL OVERRIDE: ép cứng chạy DYNAMIC (BiLSTM)")
            else:
                recognizer.force_mode = "auto"
                print("\n--> Hủy ép cứng: trở về AUTO MODE")
    
    recognizer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()