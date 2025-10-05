😍 CO2 - Time series forecasting


Dự án này sử dụng mô hình Hồi quy Tuyến tính (Linear Regression) kết hợp với kỹ thuật dự đoán trực tiếp đa bước (Direct Multi-step Forecasting) để dự đoán lượng tiêu thụ CO2 trong tương lai dựa trên dữ liệu lịch sử.
Mục đích
Áp dụng kỹ thuật xử lý dữ liệu chuỗi thời gian (Time Series) để chuẩn bị tập dữ liệu cho việc dự đoán đa bước.

Xây dựng một tập hợp các mô hình hồi quy tuyến tính độc lập, mỗi mô hình chịu trách nhiệm dự đoán một bước thời gian trong tương lai.

Đánh giá hiệu suất của các mô hình sử dụng các chỉ số thống kê tiêu chuẩn (R2, MAE, MSE).
Phương pháp Dự đoán (Direct Multi-step)
Mã nguồn sử dụng hàm create_direct_data để chuyển đổi dữ liệu chuỗi thời gian thành một tập dữ liệu giám sát (Supervised Learning).

Đầu vào (Features): Dữ liệu CO2 tại W bước thời gian gần nhất (với W=5 là window_size).
Đầu ra (Targets): Dữ liệu CO2 tại T bước thời gian tiếp theo (với T=3 là target_size).

Mô hình sẽ huấn luyện T mô hình hồi quy tuyến tính độc lập, mỗi mô hình được huấn luyện để dự đoán một mục tiêu cụ thể.

