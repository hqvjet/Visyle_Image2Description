import cv2
import easyocr

def ocr_clothing(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Khởi tạo OCR Reader
    reader = easyocr.Reader(['en', 'vi'])  # Thêm ngôn ngữ nếu cần

    # Nhận diện chữ
    results = reader.readtext(image)

    # Vẽ box và hiển thị text lên ảnh
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # In kết quả nhận diện
    return [text for (_, text, _) in results]
