# Hướng dẫn Deploy lên Render

## Tổng quan

File này hướng dẫn cách deploy AI Recommendation Server lên Render.

**Hệ thống bao gồm:**

- **Web Service**: API server cho recommendations (ai-recommendation-server)
- **Background Worker**: Scheduler tự động train model và generate recommendations (ai-scheduler)

## Yêu cầu

- Tài khoản Render (miễn phí hoặc trả phí)
- Database PostgreSQL (có thể dùng Render PostgreSQL hoặc external database)
- Models đã được train (thư mục `models/`)

## Các bước deploy

### 1. Chuẩn bị Repository

Đảm bảo code đã được push lên Git repository (GitHub, GitLab, hoặc Bitbucket).

### 2. Deploy bằng Render Blueprint (Khuyến nghị)

#### Cách 1: Sử dụng render.yaml

1. Đăng nhập vào [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Blueprint"
3. Kết nối repository chứa code
4. Render sẽ tự động detect file `render.yaml` và tạo service
5. Cấu hình các biến môi trường:

   - `DB_HOST`: Host của PostgreSQL database
   - `DB_PORT`: Port của database (thường là 5432)
   - `DB_NAME`: Tên database
   - `DB_USER`: Username database
   - `DB_PASSWORD`: Password database
   - `AI_SERVER_URL`: URL public của web service (ví dụ: `https://ai-recommendation-server.onrender.com`)

   **Lưu ý**: Sau khi web service deploy xong, copy URL và set vào `AI_SERVER_URL` cho cả web service và worker service.

#### Cách 2: Deploy thủ công

1. Đăng nhập vào [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Kết nối repository
4. Cấu hình:
   - **Name**: `ai-recommendation-server`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 ai_server:app`
5. Thêm các biến môi trường (Environment Variables):
   ```
   DB_HOST=<your_db_host>
   DB_PORT=5432
   DB_NAME=<your_db_name>
   DB_USER=<your_db_user>
   DB_PASSWORD=<your_db_password>
   MODEL_SAVE_PATH=./models
   AI_SERVER_HOST=0.0.0.0
   BATCH_SIZE=32
   EPOCHS=50
   LEARNING_RATE=0.001
   MAX_RECOMMENDATIONS=10
   RECOMMENDATION_EXPIRY_HOURS=24
   MIN_INTERACTIONS_PER_USER=5
   MIN_INTERACTIONS_PER_ITEM=3
   TRAIN_TEST_SPLIT=0.8
   COLLABORATIVE_WEIGHT=0.6
   CONTENT_BASED_WEIGHT=0.4
   ```

### 3. Tạo PostgreSQL Database trên Render (Nếu chưa có)

1. Click "New +" → "PostgreSQL"
2. Chọn plan (Free tier có giới hạn)
3. Copy connection string và cập nhật các biến môi trường

### 4. Cấu hình Scheduler (Background Worker)

Sau khi deploy web service thành công:

1. Render sẽ tự động tạo background worker service `ai-scheduler` từ `render.yaml`
2. Cấu hình `AI_SERVER_URL` cho worker service:
   - Vào worker service `ai-scheduler` trong Render Dashboard
   - Thêm environment variable: `AI_SERVER_URL=https://your-web-service.onrender.com`
   - (Thay `your-web-service` bằng tên thực tế của web service)

**Scheduler sẽ tự động:**

- Train model mỗi 6 giờ (có thể config qua `TRAINING_INTERVAL_MINUTES`)
- Generate recommendations mỗi 2 giờ (có thể config qua `RECOMMENDATION_INTERVAL_MINUTES`)
- Reload models trong web service sau khi train xong

### 5. Upload Models

Models cần được commit vào repository hoặc upload sau khi deploy:

- Đảm bảo thư mục `models/` có đầy đủ:
  - `models/encoders/user_encoder.pkl`
  - `models/encoders/item_encoder.pkl`
  - `models/encoders/scaler.pkl`
  - `models/hybrid_model/hybrid_metadata.pkl`
  - `models/hybrid_model/collaborative/collaborative_model.h5`
  - `models/hybrid_model/collaborative/user_encoder.pkl`
  - `models/hybrid_model/collaborative/item_encoder.pkl`
  - `models/hybrid_model/content_based/content_based_model.h5`

### 6. Kiểm tra Deployment

Sau khi deploy thành công:

1. Kiểm tra logs trong Render Dashboard
2. Test health endpoint: `https://your-service.onrender.com/health`
3. Test recommendations: `https://your-service.onrender.com/recommendations/anonymous?limit=5`

## API Endpoints

- `GET /health` - Health check
- `GET /recommendations/<user_id>` - Get recommendations for user
- `GET /recommendations/anonymous?limit=10&profile=general` - Get recommendations for anonymous users
- `GET /similar/<watch_id>?limit=10` - Get similar items
- `POST /reload-models` - Reload models
- `GET /stats` - Get server statistics

## Scheduler và Training

### Cấu hình Training Interval

Có thể thay đổi tần suất training và recommendation generation qua environment variables:

- `TRAINING_INTERVAL_MINUTES`: Khoảng thời gian train model (mặc định: 360 phút = 6 giờ)
- `RECOMMENDATION_INTERVAL_MINUTES`: Khoảng thời gian generate recommendations (mặc định: 120 phút = 2 giờ)

**Lưu ý**:

- Training có thể tốn nhiều tài nguyên và thời gian
- Với free tier, nên set interval lớn hơn (ví dụ: 720 phút = 12 giờ)
- Worker service cũng sẽ sleep nếu không có activity trên free tier

### Tắt Scheduler (Nếu không cần)

Nếu không muốn tự động train, có thể:

1. Xóa hoặc disable worker service `ai-scheduler` trong Render Dashboard
2. Hoặc set `TRAINING_INTERVAL_MINUTES` và `RECOMMENDATION_INTERVAL_MINUTES` thành giá trị rất lớn

## Lưu ý quan trọng

1. **Free Tier Limitations**:

   - Web service sẽ sleep sau 15 phút không có traffic
   - Worker service cũng có thể sleep nếu không có activity
   - Lần đầu wake up có thể mất 30-60 giây
   - Có giới hạn về RAM và CPU
   - **Starter plan** ($7/tháng) không bị sleep và có nhiều tài nguyên hơn

2. **Database Connection**:

   - Đảm bảo database cho phép connection từ Render IPs
   - Nếu dùng external database, cần whitelist Render IPs
   - Cả web service và worker service đều cần kết nối database

3. **Scheduler và Training**:

   - Worker service cần `AI_SERVER_URL` để reload models sau khi train
   - Training sẽ chạy tự động theo schedule đã cấu hình
   - Training lần đầu có thể mất nhiều thời gian tùy vào lượng data

4. **Models Size**:

   - Models có thể lớn, đảm bảo commit vào Git hoặc dùng persistent disk
   - Render free tier có giới hạn về disk space

5. **Environment Variables**:

   - Không commit file `.env` vào Git
   - Sử dụng Render Environment Variables để bảo mật

6. **Performance**:
   - Gunicorn được cấu hình với 2 workers và 4 threads
   - Có thể điều chỉnh dựa trên plan của bạn

## Troubleshooting

### Service không start

- Kiểm tra logs trong Render Dashboard
- Đảm bảo models đã được upload đầy đủ
- Kiểm tra database connection

### Models không load

- Kiểm tra đường dẫn `MODEL_SAVE_PATH`
- Đảm bảo tất cả file models đã được commit
- Kiểm tra logs để xem lỗi cụ thể

### Database connection failed

- Kiểm tra các biến môi trường database
- Đảm bảo database cho phép external connections
- Kiểm tra firewall rules

## Support

Nếu gặp vấn đề, kiểm tra:

1. Render Dashboard logs
2. Health endpoint response
3. Database connection status
