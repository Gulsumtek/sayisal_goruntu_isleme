import cv2
import numpy as np

# HSV Ayarını Etkileşimli Yapmak için Fonksiyon
def adjust_hsv(image):
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('H Min', 'Trackbars', 0, 180, nothing)
    cv2.createTrackbar('H Max', 'Trackbars', 31, 180, nothing)
    cv2.createTrackbar('S Min', 'Trackbars', 9, 255, nothing)
    cv2.createTrackbar('S Max', 'Trackbars', 172, 255, nothing)
    cv2.createTrackbar('V Min', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nothing)

    # İkinci trackbar penceresi
    cv2.namedWindow('Trackbars2')
    cv2.createTrackbar('H Min', 'Trackbars2', 21, 180, nothing)
    cv2.createTrackbar('H Max', 'Trackbars2', 180, 180, nothing)
    cv2.createTrackbar('S Min', 'Trackbars2', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Trackbars2', 35, 255, nothing)
    cv2.createTrackbar('V Min', 'Trackbars2', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Trackbars2', 196, 255, nothing)
    while True:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_min1 = cv2.getTrackbarPos('H Min', 'Trackbars')
        h_max1 = cv2.getTrackbarPos('H Max', 'Trackbars')
        s_min1 = cv2.getTrackbarPos('S Min', 'Trackbars')
        s_max1 = cv2.getTrackbarPos('S Max', 'Trackbars')
        v_min1 = cv2.getTrackbarPos('V Min', 'Trackbars')
        v_max1 = cv2.getTrackbarPos('V Max', 'Trackbars')

        # İkinci trackbar değerlerini al
        h_min2 = cv2.getTrackbarPos('H Min', 'Trackbars2')
        h_max2 = cv2.getTrackbarPos('H Max', 'Trackbars2')
        s_min2 = cv2.getTrackbarPos('S Min', 'Trackbars2')
        s_max2 = cv2.getTrackbarPos('S Max', 'Trackbars2')
        v_min2 = cv2.getTrackbarPos('V Min', 'Trackbars2')
        v_max2 = cv2.getTrackbarPos('V Max', 'Trackbars2')

        # HSV sınırlarını belirle
        lower_bound1 = np.array([h_min1, s_min1, v_min1])
        upper_bound1 = np.array([h_max1, s_max1, v_max1])

        lower_bound2 = np.array([h_min2, s_min2, v_min2])
        upper_bound2 = np.array([h_max2, s_max2, v_max2])

        # Maskeleri oluştur
        mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)
        mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)

        # Maskeleri birleştir
        combined_mask = cv2.bitwise_or(mask1, mask2)
        # Orijinal boyutta görüntüyü yeniden boyutlandır
        scale_factor = 0.5  # %50 oranında küçültme
        resized = cv2.resize(combined_mask, None, fx=scale_factor, fy=scale_factor)
        resized_original = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow("image",resized_original)
        cv2.imshow('Mask', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return  (lower_bound1, upper_bound1), (lower_bound2, upper_bound2)
#Belirlenen HSV Aralıkları: Lower=[97  0 52],[ 0  0 83], Upper=[180 253  82],[ 69  69 173]
#Belirlenen HSV Aralıkları: Lower=[0 9 0],[21  0  0], Upper=[ 31 172 255],[180  35 196] en iyisi!

# 2. Renk Analizi (HSV Maskeleme)
def color_analysis(image, lower_bound, upper_bound):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

# 3. Kenar Tespiti
def detect_edges(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

# 4. Maske ve Kenarları Birleştirme
def combine_results(mask, edges):
    combined = cv2.bitwise_and(mask, mask, mask=edges)
    return combined

# 5. Erozyon ve Genişletme
def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return refined

# 6. Kaçak Yapıları İşaretleme
def detect_structures(original_image, combined):
    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Alan filtresi
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_image

# Video İşleme Fonksiyonu
def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video dosyası açılamadı.")
        return

    # Videodan bir kare alarak HSV ayarı yap
    ret, sample_frame = cap.read()
    if not ret:
        print("Video boş veya bir kare alınamadı.")
        return

    # HSV sınırlarını etkileşimli olarak ayarla
    print("HSV ayarlarını yapın. Ayarlama tamamlandığında 'q' tuşuna basın.")
    sample_ss=cv2.imread("videoss.jpg")
    (lower_bound1, upper_bound1), (lower_bound2, upper_bound2) = adjust_hsv(sample_ss)
    print(f"Belirlenen HSV Aralıkları: Lower={lower_bound1},{lower_bound2}, Upper={upper_bound1},{upper_bound2}")

    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # HSV görüntüsünü oluştur
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Her kare üzerinde işlem yap
        # Her iki aralık için maskeler oluştur
        mask1 = cv2.inRange(hsv_image, np.array(lower_bound1), np.array(upper_bound1))
        mask2 = cv2.inRange(hsv_image, np.array(lower_bound2), np.array(upper_bound2))
        # Maskeleri birleştir
        combined_mask1 = cv2.bitwise_or(mask1, mask2)

        edges = detect_edges(frame)
        combined_mask2 = combine_results(combined_mask1, edges)
        refined_mask = refine_mask(combined_mask2)
        result_frame = detect_structures(frame.copy(), refined_mask)
        # Orijinal boyutta görüntüyü yeniden boyutlandır
        scale_factor = 0.5  # %50 oranında küçültme

        resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        resized_result = cv2.resize(result_frame, None, fx=scale_factor, fy=scale_factor)

        scale_factor = 0.2  # %50 oranında küçültme

        combined_mask1=cv2.resize(combined_mask1, None, fx=scale_factor, fy=scale_factor)
        edges=cv2.resize(edges, None, fx=scale_factor, fy=scale_factor)
        refined_mask=cv2.resize(refined_mask, None, fx=scale_factor, fy=scale_factor)
        # Ara adımları görselleştir
        cv2.imshow('Orijinal Kare', resized_frame)
        cv2.imshow("HSV Maske",combined_mask1)
        cv2.imshow("Canny Maske", edges)
        cv2.imshow("Birleştirilmiş", refined_mask)
        cv2.imshow('Sonuç', resized_result)

        # Videoya yaz
        if output_path:
            out.write(result_frame)

        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Ana Fonksiyon
if __name__ == "__main__":
    video_path = "ormanvideo.mp4"  # Giriş video yolu
    output_path = "output_video.avi"  # Çıkış video yolu (isteğe bağlı)

    process_video(video_path, output_path)
