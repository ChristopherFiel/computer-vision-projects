import cv2
import pytesseract

img = cv2.imread("demo.png")

img2char = pytesseract.image_to_string(img).strip()
img2box = pytesseract.image_to_boxes(img)

# print(img2char)
print(img2box)
print(img.shape)

imgH, imgW, _ = img.shape


for boxes in img2box.splitlines():
    boxes = boxes.split(" ")
    x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
    cv2.rectangle(img, (x, imgH - y), (w, imgH - h), (0, 0, 255), 2)

cv2.putText(
    img,
    img2char,
    (x // 2, y),
    cv2.FONT_HERSHEY_PLAIN,
    2,
    (255, 0, 0),
    4,
)
cv2.imwrite("outlined.png", img)
cv2.imshow("demo", img)
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows
