# Hướng dẫn chạy dự án Multi-modal Defect Detection

Dự án này sử dụng mô hình học sâu để phân tích và nhận diện lỗi, được viết dưới dạng Jupyter Notebook. Các bước để cài đặt môi trường và chạy mã nguồn.

## Cài đặt thư viện
1. Cài đặt các thư viện cần thiết cho việc xử lý ảnh, dữ liệu và mô hình AI:
   ```bash
   pip install torch torchvision transformers opencv-python matplotlib albumentations numpy requests tqdm
   ```

## Hướng dẫn chạy

1. Mở Terminal (hoặc Command Prompt) tại cùng thư mục dự án `multi_modal_defect_detection`.

chạy `multi_modal_defect_detection_v2.ipynb`.
data train :https://drive.google.com/file/d/1qSzLlENR7uLomJ2uvNtxICkIEYPVdDVl/view?usp=sharing
bỏ data train cùng cấp với file ipynb 
. chạy từng step và không cần chạy step 5,6 vì đã train sẵn 
## 2. Features (Tính năng hệ thống)
*   **Conveyor Object Tracking:** Theo dõi và đếm số lượng đai ốc chạy trên băng chuyền mà không bị đếm trùng lặp.
*   **Real-time Defect Classification:** Phân loại ngay lập tức đồ vật đó. Cụ thể có 4 loại lỗi và 1 trạng thái tốt:
    *   *Scratch:* Bề mặt bị trầy xước.
    *   *Bent:* Đai ốc bị móp méo, uốn cong hoặc biến dạng hỏng hóc.
    *   *Color:* Bề mặt bị đổi màu, rỉ sét hạt.
    *   *Flip:* Đai ốc bị lật ngược mặt (ngược chiều).
    *   *Good:* Đi qua bài test, chất lượng đạt chuẩn.
*   **Localization & Masking:** Tô đỏ trực tiếp lên bộ phận bị lỗi trên màn hình giám sát để công nhân dễ nhìn thấy.
*   **Visual Question Answering (VQA):** Cho phép đặt câu hỏi  và trả lời (VD: "scratch", "bent").
*   **Automated Report Logging:** Tự động lưu và xuất báo cáo CSV

## 3. Tech Solutions (Giải pháp Kỹ thuật)
**Dataset (Dữ liệu chuẩn):**
*   MVTec AD (Tập con đồ vật: metal_nut segmentation).

**Mô hình Phát hiện (Detection Model):**
*   CenterNet-style BBox Head.

**Mô hình Phân vùng (Segmentation Model):**
*   U-Net.

**Multi-modal AI (AI Đa phương thức):**
*   *Image Encoder:* Nhóm mô hình CNN (ResNet-18 rút gọn).
*   *Text Encoder:* Nhóm mô hình Transformer (DistilBERT).
*   *Fusion Modules:* Cơ chế Attention chéo (Cross-Attention).



## 4. Thiết kế luồng thuật toán (Logic + AI)

Dự án là sự kết hợp chặt chẽ giữa 2 tầng thuật toán: Tầng Logic (ứng dụng) và Tầng AI (bộ não học sâu).

### 4.1. Phần Logic 
Đảm nhận vai trò xử lý thế giới vật lý thực, xác định vị trí toạ độ và dựng giao diện:
1. **Trích xuất vật thể (Image Processing):** Đưa video Camera qua hệ ảnh Xám (Grayscale) ➔ Làm mờ (Gaussian) ➔ Tách biến (Canny Edge) ➔ Khử nhiễu Cạnh (Dilate) ➔ Thuật toán bao lồi (Find Contours) để chủ động cắt rời khung hình đai ốc (ROI) ra khỏi băng chuyền nền đen.
2. **Thuật toán Tracking (OpenCV Centroid Tracking):** Tính toán Toạ độ tâm `(cx, cy)` của vùng đai ốc. Dùng định lý hình học Euclidean để tính khoảng cách dịch chuyển giữa các frame. Nếu khoảng cách rất nhỏ ➔ Hệ thống tự hiểu nó là cùng 1 vật thể ➔ Gán **ID duy nhất** (Nhằm giải quyết bài toán chống đếm trùng lặp).
3. **Logic Render & Cảnh báo:** Đón nhận kết quả từ mạng AI trả về. Nhận màng phân vùng lỗi nhỏ (Mask) từ U-Net AI ➔ Ghi đè ma trận màu đỏ `(255,0,0)` lên Mask ➔ Cập nhật đè lên khung hình video với độ mờ 50% bằng hàm `cv2.addWeighted` ➔ Viết logic Python trích xuất thông số (Tỷ lệ lỗi, Loại lỗi) đẩy ra tệp `BaoCao.csv`.

### 4.2. Phần AI 
Đóng vai trò là "Bộ Não" phân tích nhận diện hình dạng lỗi và tạo ngữ nghĩa để trả lời câu hỏi tự động.

**[1] Nhánh Hình ảnh (Vision Pipeline):**
`Ảnh hạt đai ốc (ROI)` ➔ `Tiền xử lý (Resize/Normalize)` ➔ `ResNet-18 (Backbone)` ➔ **`Visual Feature Map`**

Từ **`Visual Feature Map`**, mô hình tách thành 3 luồng để giải quyết 3 nhiệm vụ Computer Vision song song:
* Luồng 1 ➔ `CenterNet Head` ➔ *Đầu ra:* **Bounding Box & Xác suất mắc lỗi**
* Luồng 2 ➔ `U-Net Encoder` ➔ *Đầu ra:* **Defect Mask** (Ma trận vẽ bản đồ khuôn dáng vị trí lỗi)
* Luồng 3 ➔ `Global Average Pooling` ➔ **`Visual Embedding`** (Nén đặc trưng hình ảnh thành đại số)

**[2] Nhánh Văn bản (NLP Pipeline):**
`Câu hỏi truy vấn (Text)` ➔ `DistilBERT Encoder` ➔ **`Text Embedding`** (Nén đặc trưng ngữ nghĩa chữ thành đại số)

**[3] Khối Trộn Trung tâm (Multi-modal Fusion):**
Áp dụng học Đa tác vụ ghép nối chéo (`Visual Embedding` + `Text Embedding`) ➔ Đẩy vào mô-đun `Cross-Attention Fusion Module` ➔ Đẩy qua `VQA Classifier Head` ➔ *Đầu ra:* **Câu trả lời Văn bản (VQA Answer, vd: scratch, bent, good...)**