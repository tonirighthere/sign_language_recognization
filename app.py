# ============================================
# REAL-TIME HYBRID ASL RECOGNITION
# ============================================

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
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Load CNN model (cho static)
        self.cnn_model = self._load_cnn_model(cnn_model_path)
        
        # Load BiLSTM model (cho dynamic)
        self.bilstm_model = self._load_bilstm_model(bilstm_model_path)
        
        # Frame buffer cho dynamic recognition
        self.frame_buffer = deque(maxlen=seq_length)
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
        
        # Hand connections
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Smoothing
        self.prediction_buffer = deque(maxlen=5)
        
        print(f"✅ Hybrid recognizer initialized")
        print(f"   Device: {self.device}")
        print(f"   Mode: Static (CNN) + Dynamic (BiLSTM) for J/Z")
    
    def _load_cnn_model(self, model_path):
        """Load CNN model cho static recognition"""
        class LightweightASL_CNN(nn.Module):
            def __init__(self, num_classes=29, img_size=224):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                )
                self._to_linear = 128 * (img_size // 8) * (img_size // 8)
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(self._to_linear, 256), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
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
        
        class BiLSTM_Attention(nn.Module):
            def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=29):
                super().__init__()
                self.bn = nn.BatchNorm1d(input_size)
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                    batch_first=True, bidirectional=True, dropout=0.3)
                self.attention = AdditiveAttention(hidden_size * 2)
                self.fc1 = nn.Linear(hidden_size * 2, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                batch, seq, feat = x.shape
                x = self.bn(x.view(-1, feat)).view(batch, seq, feat)
                lstm_out, _ = self.lstm(x)
                context, _ = self.attention(lstm_out)
                
                out = self.dropout(context)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return out
        
        model = BiLSTM_Attention(num_classes=self.num_classes)
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
            if connection[0] < len(points) and connection[1] < len(points):
                cv2.line(img, points[connection[0]], points[connection[1]], (255, 255, 255), 2)
        
        for pt in points:
            cv2.circle(img, pt, 3, (0, 255, 0), -1)
        
        return img
    
    def extract_keypoints(self, hand_landmarks):
        """Trích xuất keypoints từ hand landmarks"""
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        # Pad to 63 features (21x3)
        while len(keypoints) < 63:
            keypoints.append(0.0)
        return np.array(keypoints[:63])
    
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
        """Dự đoán bằng CNN (cho ký tự tĩnh)"""
        img = cv2.resize(bone_img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        # Bước cực kỳ quan trọng: NORMALIZE giống hệt lúc train!
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.cnn_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
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
            
            if is_moving:
                # Đang có chuyển động -> dùng BiLSTM
                self.dynamic_counter += 1
                self.static_counter = 0
                
                if self.dynamic_counter > 5:  # Giảm độ trễ
                    prediction, confidence = self.predict_dynamic()
                    self.current_mode = "dynamic"
            else:
                # Tĩnh -> dùng CNN
                self.static_counter += 1
                self.dynamic_counter = 0
                
                if self.static_counter > 2:  # Giảm độ trễ
                    prediction, confidence = self.predict_static(bone_img)
                    self.current_mode = "static"
            
            # Lưu vào buffer để làm mượt (tránh nháy chữ)
            if prediction is not None:
                self.prediction_buffer.append((prediction, confidence))
                
            # Trích xuất kết quả mượt từ buffer thay vì kết quả tức thời
            if len(self.prediction_buffer) > 0:
                from collections import Counter
                classes = [p[0] for p in self.prediction_buffer]
                most_common = Counter(classes).most_common(1)[0][0]
                confs = [p[1] for p in self.prediction_buffer if p[0] == most_common]
                
                prediction = most_common
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
        if prediction:
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