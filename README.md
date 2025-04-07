# STT Integration LLM

**STT Integration LLM** là một dự án tích hợp mô hình nhận diện giọng nói (Speech-to-Text - STT) với mô hình ngôn ngữ lớn (Large Language Model - LLM) để nâng cao khả năng xử lý và chuyển đổi âm thanh thành văn bản tiếng Việt. Dự án sử dụng các mô hình tiên tiến như **PhoWhisper-large** của VinAI và **Viet-Mistral-7B-Chat** để đạt hiệu suất cao trong việc nhận diện và chuyển đổi giọng nói.

## Kết quả huấn luyện
Kết quả huấn luyện của dự án có thể được theo dõi tại:  
[**W&B Run**](https://wandb.ai/huutri231103/slam-llm/runs/9iic653m?nw=nwuserhuutri231103)

## Lời cảm ơn
Dự án này được xây dựng dựa trên nền tảng của các công trình mã nguồn mở tuyệt vời sau:  
- [X-LANCE/SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)  
- [Viet-Mistral/Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)  
- [VinAI/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large)  

Chúng tôi xin gửi lời cảm ơn chân thành đến các tác giả và cộng đồng đã đóng góp vào các dự án này.

## Giấy phép
Dự án được phát hành dưới **Giấy phép MIT**. Xem chi tiết trong tệp [LICENSE](LICENSE).

## Cấu trúc thư mục
Dưới đây là cấu trúc thư mục của dự án:

```
fuutoru-stt-integration-llm/
├── README.md                   # Tài liệu hướng dẫn chính
├── LICENSE                     # Giấy phép MIT
├── scripts/                    # Tập lệnh chính
│   ├── config.py               # Cấu hình mô hình, huấn luyện và dữ liệu
│   ├── finetune.py             # Tập lệnh huấn luyện mô hình
│   ├── inference.py            # Tập lệnh suy luận (inference)
│   ├── finetune/               # Tập lệnh shell cho huấn luyện
│   │   └── finetune_phowhisper_large_vietmistral_7b.sh
│   └── inference/              # Tập lệnh shell cho suy luận
│       └── inference_phowhisper_large_vietmistral_7b.sh
└── src/                        # Mã nguồn chính
    ├── data/                   # Xử lý dữ liệu
    │   ├── concatenator.py     # Ghép nối dữ liệu
    │   └── sampler.py          # Lấy mẫu dữ liệu
    ├── dataset/                # Tập dữ liệu
    │   └── speech_dataset.py   # Xử lý tập dữ liệu âm thanh
    ├── models/                 # Các mô hình
    │   ├── encoder.py          # Mã hóa âm thanh (Whisper)
    │   ├── projector.py        # Chiếu dữ liệu từ encoder sang LLM
    │   └── stt_model.py        # Mô hình STT tích hợp
    ├── pipeline/               # Quy trình huấn luyện và suy luận
    │   ├── finetune.py         # Pipeline huấn luyện
    │   └── inference_batch.py  # Pipeline suy luận hàng loạt
    ├── policies/               # Chính sách tối ưu hóa
    │   ├── activation_checkpointing_functions.py
    │   ├── anyprecision_optimizer.py
    │   ├── mixed_precision.py
    │   └── wrapping.py
    └── utils/                  # Tiện ích hỗ trợ
        ├── checkpoint_handler.py
        ├── dataset_utils.py
        ├── memory_utils.py
        ├── metric.py
        ├── model_utils.py
        └── train_utils.py
```

## Yêu cầu cài đặt
Để chạy dự án, bạn cần cài đặt các thư viện sau:
- Python 3.8+
- PyTorch (phiên bản hỗ trợ CUDA nếu dùng GPU)
- Transformers (Hugging Face)
- Hydra (quản lý cấu hình)
- Wandb (theo dõi huấn luyện, tùy chọn)
- Các thư viện khác được liệt kê trong `requirements.txt` (nếu có).

Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Dữ liệu đầu vào cần ở định dạng `.jsonl`, với mỗi dòng chứa thông tin về đường dẫn âm thanh (`source`) và văn bản mục tiêu (`target`). Ví dụ:
```json
{"source": "/path/to/audio.wav", "target": "Xin chào, đây là một ví dụ."}
```

### 2. Huấn luyện mô hình
Sử dụng tập lệnh shell để huấn luyện:
```bash
cd scripts/finetune
bash finetune_phowhisper_large_vietmistral_7b.sh
```
- Cập nhật các đường dẫn trong tập lệnh (`speech_encoder_path`, `llm_path`, `train_data_path`, `val_data_path`, v.v.) theo hệ thống của bạn.
- Điều chỉnh các tham số trong `hydra_args` nếu cần.

### 3. Suy luận (Inference)
Sử dụng tập lệnh shell để suy luận:
```bash
cd scripts/inference
bash inference_phowhisper_large_vietmistral_7b.sh
```
- Cập nhật các đường dẫn (`speech_encoder_path`, `llm_path`, `val_data_path`, `ckpt_path`) và các tham số khác theo nhu cầu.

### 4. Theo dõi huấn luyện
Nếu bật `use_wandb` trong `log_config`, bạn có thể theo dõi quá trình huấn luyện trên Weights & Biases. Đảm bảo cung cấp `WANDB_API_KEY` trong tập lệnh shell.

## Cấu hình chính
Tệp `config.py` chứa các cấu hình quan trọng:
- `ModelConfig`: Cấu hình mô hình (encoder, LLM, projector).
- `TrainConfig`: Tham số huấn luyện (epochs, batch size, learning rate, v.v.).
- `DataConfig`: Cấu hình dữ liệu (đường dẫn, định dạng đầu vào).
- `FSDPConfig`: Cấu hình phân mảnh dữ liệu song song (nếu dùng FSDP).
- `LogConfig`: Cấu hình ghi log và theo dõi.

## Đóng góp
Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:
1. Fork dự án.
2. Tạo nhánh mới (`git checkout -b feature/your-feature`).
3. Commit thay đổi (`git commit -m "Thêm tính năng XYZ"`).
4. Push lên nhánh của bạn (`git push origin feature/your-feature`).
5. Tạo Pull Request.

## Liên hệ
Nếu bạn có câu hỏi hoặc cần hỗ trợ, vui lòng liên hệ qua email: [huutri231103@gmail.com](mailto:huutri231103@gmail.com).
