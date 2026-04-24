# Hướng dẫn chạy dự án Multi-modal Defect Detection

Dự án này sử dụng mô hình học sâu để phân tích và nhận diện lỗi, được viết dưới dạng Jupyter Notebook. Dưới đây là các bước để cài đặt môi trường và chạy mã nguồn.

## Yêu cầu hệ thống
- **Ngôn ngữ**: Python 3.8+
- **Phần cứng**: Khuyên dùng thiết bị có hỗ trợ GPU (NVIDIA/CUDA) để tăng tốc độ huấn luyện và dự đoán mô hình. Nếu chạy bằng CPU, thời gian chờ sẽ khá lâu.

## Cài đặt thư viện

1. Nếu bạn chưa có Jupyter Notebook, hãy cài đặt qua pip:
   ```bash
   pip install jupyter
   ```

2. Cài đặt các thư viện cần thiết cho việc xử lý ảnh, dữ liệu và mô hình AI:
   ```bash
   pip install torch torchvision transformers opencv-python matplotlib albumentations numpy requests tqdm
   ```

## Dữ liệu và Mô hình
- Dự án đòi hỏi phải có thư mục data/dataset và trọng số mô hình đã huấn luyện.
- Hiện tại trong thư mục đã có file `best_model.pth` và thư mục `dataset`. Đảm bảo những file này nằm cùng cấu trúc thư mục với tệp `.ipynb`. 
- Nếu thấy thiếu dữ liệu, hãy giải nén file `multi_modal_defect_detection.rar`.

## Hướng dẫn chạy

1. Mở Terminal (hoặc Command Prompt) tại cùng thư mục dự án `multi_modal_defect_detection`.
2. Khởi chạy Jupyter bằng lệnh:
   ```bash
   jupyter notebook
   ```
3. Trình duyệt web sẽ được tự động mở ra. Bạn tiếp tục bấm chọn file `multi_modal_defect_detection_v2.ipynb`.
4. Trên thanh công cụ của Jupyter, bấm chọn **Run > Run All Cells** (hoặc **Cell > Run All**) để tuần tự chạy toàn bộ bộ nhận diện - từ việc nạp cấu hình, mô hình, xử lý đánh giá dữ liệu.
