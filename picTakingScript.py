import cv2
import os
import time
import uuid

def create_dataset(number_imgs):
    images_path = 'dataset'
    labels = ['a', 'b','c','d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    #'a', 'b','c','d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'

    cTime = 0
    pTime = 0
    for label in labels: 
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print('Collecting images for {}'.format(label))
        time.sleep(3)
        for imgnum in range(number_imgs):
            cTime = time.time()
            pTime = cTime
            imagename = os.path.join(images_path, label, label+'{}.jpg'.format(imgnum + 1))
            while pTime < cTime + .01:
                success, img = cap.read()
                cv2.imshow('{}'.format(label), img)
                pTime = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            print(imgnum)
            if imgnum != 0:
                cv2.imwrite(imagename, img)
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    create_dataset()

