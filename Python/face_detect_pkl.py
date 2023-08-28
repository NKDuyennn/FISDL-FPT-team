import face_recognition
import os
import pickle
import numpy as np

# Đường dẫn tới thư mục chứa ảnh train
images_folder = "/content/drive/MyDrive/faceid_iotchallenge/FISDL-FPT-team/Python/esp-img/"

train_face_encodings = []

# Lặp qua tất cả các tệp tin trong thư mục
for image_file in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_file)
    
    # Nạp ảnh
    image = face_recognition.load_image_file(image_path)
    
    k_face_locations = face_recognition.face_locations(image)
    # Tìm và mã hóa khuôn mặt trong ảnh
    k_face_encodings = face_recognition.face_encodings(image,k_face_locations)
    
    # Kiểm tra xem có khuôn mặt trong ảnh hay không
    if len(k_face_encodings) > 0:
        train_face_encodings.append(k_face_encodings[0])

# Lưu danh sách các vector biểu diễn đã train vào tệp tin
model = "trained_face_encodings.pkl"
with open(model, "wb") as f:
    pickle.dump(train_face_encodings, f)

print("Training completed.")

# Nạp danh sách vector biểu diễn từ tệp .pkl
with open(model, "rb") as f:
    known_face_encodings = pickle.load(f)

# Chuyển đổi từ mảng NumPy sang danh sách

# Load bức ảnh cần nhận dạng
unknown_image = face_recognition.load_image_file("/content/drive/MyDrive/faceid_iotchallenge/FISDL-FPT-team/Python/static/tuyen.jpg")

# Tìm các khuôn mặt trong ảnh
unknown_locations = face_recognition.face_locations(unknown_image)
unknown_encodings = face_recognition.face_encodings(unknown_image, unknown_locations)

# So sánh khuôn mặt trong ảnh với danh sách vector biểu diễn đã train
matches = face_recognition.compare_faces(np.array(known_face_encodings), unknown_encodings)

print(matches)