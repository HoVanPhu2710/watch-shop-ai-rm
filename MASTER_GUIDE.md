# AI Recommendation System - Master Guide

## 🎯 Tổng quan hệ thống

### **Kiến trúc:**

```
Main API (Port 3000) → AI Server (Port 5001) → Real-time ML Models → Fresh Recommendations
```

### **Database:**

- ✅ **Giữ lại**: `users`, `watches`, `user_interactions` + business tables
- ❌ **Đã xóa**: `recommendations`, `model_training_history` tables

## 🚀 Hướng dẫn Setup & Deploy

### **Step 1: Train Model (Một lần)**

```bash
cd ai-recommend
pip install -r requirements_ai_server.txt
python train_model_fixed.py
```

### **Step 2: Start AI Server**

```bash
# Linux/Mac
./setup_and_start.sh

# Windows
setup_and_start.bat

# Hoặc manual
python ai_server.py
```

### **Step 3: Start Main API**

```bash
cd watch-shop-be
npm start
```

## 📡 API Endpoints

### **🎯 Recommendations:**

- `GET /api/v1/recommendations/recommendations/:userId` - Lấy đề xuất cho user (mặc định bao gồm cả đồng hồ đã tương tác)
- `GET /api/v1/recommendations/similar/:watchId` - Lấy sản phẩm tương tự
- `POST /api/v1/recommendations/interactions` - Ghi nhận tương tác

### **🔧 Monitoring:**

- `GET /api/v1/recommendations/ai/health` - Kiểm tra AI server
- `GET /api/v1/recommendations/ai/stats` - Thống kê AI server
- `GET /api/v1/recommendations/stats` - Thống kê tổng quan

## 📊 Example Usage

### **Lấy đề xuất:**

```bash
curl "http://localhost:3000/api/v1/recommendations/recommendations/1?limit=5"
```

### **Ghi nhận tương tác:**

```bash
curl -X POST "http://localhost:3000/api/v1/recommendations/interactions" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"watch_id":123,"interaction_type":"view"}'
```

### **JavaScript Frontend:**

```javascript
// Lấy đề xuất
async function getRecommendations(userId, limit = 10) {
  const response = await fetch(
    `http://localhost:3000/api/v1/recommendations/recommendations/${userId}?limit=${limit}`
  );
  return await response.json();
}

// Ghi nhận tương tác
async function recordInteraction(userId, watchId, interactionType) {
  const response = await fetch("/api/v1/recommendations/interactions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      watch_id: watchId,
      interaction_type: interactionType,
      session_id: Date.now().toString(),
    }),
  });
  return await response.json();
}
```

## 🔄 Lập lịch Training

### **Manual Retraining:**

```bash
cd ai-recommend
python train_model_fixed.py
```

### **Auto Schedule (Cron):**

```bash
# Retrain hàng ngày lúc 2 AM
0 2 * * * cd /path/to/ai-recommend && python train_model_fixed.py
```

## 📁 File Structure (Sau khi gom gọn)

```
ai-recommend/
├── ai_server.py                 # AI server chính
├── train_model_fixed.py         # Train model (fixed version)
├── data_processor_fixed.py      # Data processor (fixed version)
├── setup_and_start.sh           # Setup script (Linux/Mac)
├── setup_and_start.bat          # Setup script (Windows)
├── requirements_ai_server.txt   # Python dependencies
├── MASTER_GUIDE.md              # File này - hướng dẫn tổng hợp
└── models/                      # Trained models

watch-shop-be/
├── src/services/
│   └── ai-recommendation.service.js  # AI service client
├── src/controllers/
│   └── recommendation.controller.js  # Updated controller
└── API_RECOMMENDATIONS.md       # API documentation
```

## 🎯 Benefits

### **Performance:**

- ⚡ **Real-time**: No DB queries for recommendations
- 🚀 **Fast**: In-memory model inference
- 📊 **Scalable**: Independent AI server

### **Database:**

- 💾 **Lightweight**: No recommendation storage
- 🔄 **Fresh**: Always latest model predictions
- 🛠️ **Clean**: Removed unused tables

## 🔍 Troubleshooting

### **AI Server không start:**

```bash
# Check logs
tail -f ai_server.log

# Check health
curl http://localhost:5001/health
```

### **Models không load:**

```bash
# Retrain models
python train_model_fixed.py

# Check model files
ls -la models/hybrid_model/
```

### **API không hoạt động:**

```bash
# Check main API
curl http://localhost:3000/api/v1/recommendations/ai/health

# Check AI server
curl http://localhost:5001/health
```

## ✅ Migration Complete

Hệ thống đã được tối ưu hóa:

- ❌ **Xóa**: Các file cũ, documentation trùng lặp
- ✅ **Gộp**: Tất cả hướng dẫn vào 1 file
- 🚀 **Sẵn sàng**: Deploy và sử dụng ngay

**Chỉ cần 3 bước: Train → Start AI Server → Start Main API!** 🎉
