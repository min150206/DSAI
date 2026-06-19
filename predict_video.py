def predict_video():
    """Predict on video file with bounding boxes."""
    from ultralytics import YOLO
    import cv2

    WEIGHTS  = "./System/train_yolo11_final/weights/best.pt"
    VIDEO_IN = "./Demo/datatest/demo_video.mp4"
    CONF     = 0.5
    DELAY    = 5   # the larger the slower th video is(ms), 1=fastest, 60=very slow

    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(VIDEO_IN)

    #Resolution
    DISPLAY_W = 1280
    DISPLAY_H = 720

    print("Playing... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize trước khi predict
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        results   = model.predict(frame, conf=CONF, verbose=False, device = 0)[0]  #Nếu k có gpu nvidia thì bỏ "device = 0" đi nha
        annotated = results.plot()

        cv2.imshow("GTSRB Detection", annotated)
        if cv2.waitKey(DELAY) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


predict_video()