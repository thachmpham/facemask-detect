# Toán Cho Trí Tuệ Nhân Tạo - Đề Tài Nhận Diện Khẩu Trang

## Giới Thiệu

Trong project này, chúng ta sẽ xây dựng model có khả năng phân biệt hai trường hợp: người trong ảnh có đeo khẩu trang và người trong ảnh không đeo khẩu trang. Project vận dụng một số phương pháp như Convolution Neural Network, Linear Regression, Logistic Regression và Backpropagation. 

## Chuẩn bị dữ liệu huấn luyện

Tập data sử dụng để train và đánh giá model được lấy từ [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). 
Tập data gồm hai loại: 

- Ảnh người đeo khẩu trang

- Ảnh người không đeo khẩu trang

Tuy nhiên, ngoài khuôn mặt, ảnh trong tập data chứa nhiều thành phần khác như background, vai, mũ nón... Vì vậy, ta sẽ sử dụng pretrain model phát hiện khuôn mặt để tạo một tập data mới chỉ chứa khuôn mặt từ tập data cũ.

Để download và trích xuất data, ta sử dụng những lệnh sau đây.

```
kaggle datasets download -d omkargurav/face-mask-dataset
unzip face-mask-dataset.zip
mv data/with_mask data/1
mv data/without_mask data/0
python3 extract_face.py
```

Quá trình extract data còn được mô tả ở file `extract_face.ipynb`

## Xác định cấu trúc model

Mạng Convolutional Neural Network (CNN) được ứng dụng rộng rãi trong các bài toán xử lý ảnh. Trong bài này, chúng ta sẽ xây dựng một mạng CNN với cấu trúc như hình bên dưới. Lớp input nhận vào một ảnh có 3 channel và kích thước 32 x 32. Kích thước này được tham khảo từ Kaggle blog. Vì đặc trưng khẩu trang khá rõ ràng, nên dù kích thước input nhỏ, nhưng model vẫn cho kết quả chấp nhận được. Tại hidden layer, ta kết hợp lớp convolution và lớp max pool để làm giảm kích thước của ảnh mà vẫn giữ lại thông tin của bức ảnh nhiều nhất. Lớp flatten giúp chuyển data trên các convolution filter từ dạng đa chiều về một chiều. Sau lớp flatten, bức ảnh có kích thước 3 x 32 x 32 đã được chuyển đổi về dạng mảng 1 chiều của 300 đặc trưng. Tại đây, ta kết hợp linear regression và logistic regression để đưa output về 1 con số duy nhất thể hiện xác suất ảnh đó có khẩu trang hay không.

<img src="architecture.png"/>

## Training và đánh giá model

Train dataset có 5515 ảnh. Vì số lượng ảnh lớn nên ta chia train dataset thành các batch nhỏ, mỗi batch chứa 4 ảnh. Và thực hiện train lần lượt trên các batch.

Model được train với khoảng 20 epoch và learning rate là 0.001. Hàm loss được sử dụng là Binary Cross Entropy. Phương pháp được sử dụng để tối ưu bộ trọng số là Stochastic Gradient Descent.

Để train và xem kết quả đánh giá model, ta dùng lệnh sau đây.  
`python3 facemask_model.py`

Quá trình train model còn được mô tả ở file `facemask_model.ipynb`.

## Hướng dẫn sử dụng inference module

Model trả về một số p trong khoảng [0, 1] thể hiện xác suất người trong ảnh có đeo khẩu trang hay không. Số càng lớn thì khả năng người đó đeo khẩu trang càng cao.
Để đưa đến người dùng dự đoán cuối cùng, ta sử dụng phương pháp chọn số nguyên gần nhất với xác suất model đưa ra, cụ thể trong trường hợp xác suất p lớn hơn hoặc bằng 0.5, ta dự đoán người trong ảnh có đeo khẩu trang. Và ngược lại, khi xác suất nhỏ hơn 0.5, ta dự đoán người trong ảnh không đeo khẩu trang. Đoạn code sau đây mình hoạ cách sử dụng model.  

```

from facemask_model import *
import PIL

model = FacemaskRecognizeModel()
load_pretrain(model, 'pretrain/mask-reg.pth')

img = PIL.Image.open('input.png')
is_mask_exist = predict(model, img) 

# giải thích kết quả trả về
# 0: không có khẩu trang
# 1: có khẩu trang
```

## Nguồn tham khảo
- [Face Mask Detection Kaggle Notebook](https://www.kaggle.com/code/charlessamuel/face-mask-detection-pytorch)


## Thành viên thực hiện
- Phạm Minh Thạch - 22C15018


