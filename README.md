# AI Recommendation System

Hệ thống recommendation sử dụng Hybrid Model (Collaborative Filtering + Content-Based Filtering).

## Cấu trúc Project

```
ai-recommend/
├── src/                          # Source code chính
│   ├── ai_server.py              # Flask API server
│   ├── scheduler.py              # Background scheduler cho training
│   ├── train_model_fixed.py      # Script training model
│   ├── database.py                # Database connection
│   ├── config.py                 # Configuration
│   ├── data_processor_fixed.py   # Data preprocessing
│   ├── hybrid_model.py           # Hybrid recommendation model
│   ├── collaborative_filtering.py # Collaborative filtering model
│   └── content_based_filtering.py # Content-based filtering model
│
├── scripts/                       # Utility scripts
│   ├── plot_training_results.py   # Script tạo training plots
│   └── training_plots/           # Output folder cho plots
│
├── docs/                          # Documentation
│   ├── README_RENDER.md           # Hướng dẫn deploy lên Render
│   ├── MASTER_GUIDE.md            # Master guide
│   ├── add_sample_watches.sql     # Sample SQL data
│   └── env_example.txt            # Environment variables example
│
├── models/                        # Trained models (không commit vào Git)
│   ├── encoders/                  # Encoders (user, item, scaler)
│   └── hybrid_model/              # Hybrid model files
│
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render deployment config
├── Procfile                       # Procfile cho Render
└── .renderignore                  # Files to ignore khi deploy

```

## Cài đặt

1. Clone repository
2. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Tạo file `.env` từ `docs/env_example.txt` và cấu hình database
4. Train model:
   ```bash
   python src/train_model_fixed.py
   ```
5. Chạy server:
   ```bash
   python src/ai_server.py
   ```
6. Chạy scheduler (optional):
   ```bash
   python src/scheduler.py
   ```

## API Endpoints

- `GET /health` - Health check
- `GET /recommendations/<user_id>` - Get recommendations for user
- `GET /recommendations/anonymous?limit=10&profile=general` - Get recommendations for anonymous users
- `GET /similar/<watch_id>?limit=10` - Get similar items
- `POST /reload-models` - Reload models
- `GET /stats` - Get server statistics

## Deploy lên Render

Xem hướng dẫn chi tiết trong [docs/README_RENDER.md](docs/README_RENDER.md)

## Development

- Source code chính: `src/`
- Scripts utility: `scripts/`
- Documentation: `docs/`

