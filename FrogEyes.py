import multiprocessing as mp
import cv2
import numpy as np

'''Yonv1943 2018-06-03 1539'''
"""Yonv1943 2018-06-08 1142 stable"""


class Global(object):
    user_name, user_pwd = "admin", "!QAZ2wsx3edc"

    camera_ip_l = [
        # "192.168.1.164",
        # "192.168.1.165",
        # "192.168.1.166",
        # "192.168.1.167",
        # "192.168.1.168",
        # "192.168.1.169",
        "192.168.1.170",
    ]

    pass


G = Global()


class FrogEye(object):
    def __init__(self, img):
        self.min_thresh = 56.0

        img = self.img_preprocessing(img)
        background_change_after_read_image_number = 64  # change time = 0.04*number = 2.7 = 0.4*64
        self.img_list = [img for _ in range(background_change_after_read_image_number)]

        self.img_len0 = int(360)
        self.img_len1 = int(self.img_len0 / (img.shape[0] / img.shape[1]))
        self.img_back = self.img_preprocessing(img)  # background

        self.min_side_num = 3
        self.min_side_len = int(self.img_len0 / 24)  # min side len of polygon
        self.min_poly_len = int(self.img_len0 // 1.5)
        self.thresh_blur = int(self.img_len0 / 8)

    def img_preprocessing(self, img):
        img = np.copy(img)
        img = cv2.blur(img, (5, 5))
        # img = cv2.resize(img, (self.img_len1, self.img_len0))
        # img = cv2.bilateralFilter(img, d=3, sigmaColor=16, sigmaSpace=32)
        return img

    def get_polygon_contours(self, img, img_back):
        # img = np.copy(img)
        dif = np.array(img, dtype=np.int16)
        dif = np.abs(dif - img_back)
        dif = np.array(dif, dtype=np.uint8)  # get different

        gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self.min_thresh, 255, 0)
        thresh = cv2.blur(thresh, (self.thresh_blur, self.thresh_blur))

        if np.max(thresh) == 0:  # have not different
            contours = []
        else:
            thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # hulls = [cv2.convexHull(cnt) for cnt, hie in zip(contours, hierarchy[0]) if hie[2] == -1]
            # hulls = [hull for hull in hulls if cv2.arcLength(hull, True) > self.min_hull_len]
            # contours = hulls

            approxs = [cv2.approxPolyDP(cnt, self.min_side_len, True) for cnt in contours]
            approxs = [approx for approx in approxs
                       if len(approx) > self.min_side_num and cv2.arcLength(approx, True) > self.min_poly_len]
            contours = approxs
        return contours

    def main_get_img_show(self, origin_img):
        img = self.img_preprocessing(origin_img)
        contours = self.get_polygon_contours(img, self.img_back)

        self.img_list.append(img)
        img_prev = self.img_list.pop(0)

        self.img_back = img \
            if not contours or not self.get_polygon_contours(img, img_prev) \
            else self.img_back

        # hight_light = np.zeros(img.shape, dtype=np.uint8) + 2
        # hight_light = cv2.fillPoly(hight_light, contours, (1, 1, 1))
        # show_img = origin_img // hight_light

        # show_img = cv2.polylines(origin_img, contours, True, (0, 0, 255), 2)
        # return show_img
        return contours


def queue_img_get(q, window_name):
    (is_opened, frame) = q.get()

    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    frog_eye = FrogEye(frame)

    '''loop'''
    while is_opened:
        (is_opened, frame) = q.get()
        img_show = frog_eye.main_get_img_show(frame)
        cv2.imshow(window_name, img_show)
        cv2.waitKey(1)
    cv2.destroyWindow(window_name)


def queue_img_put(q, name, pwd, ip, channel=3):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    is_opened, frame = cap.read()

    '''loop'''
    while is_opened:
        is_opened, frame = cap.read()
        q.put([is_opened, frame])
        q.get() if q.qsize() > 1 else None
    else:
        print("||| camera close:", ip)


def run():
    mp.set_start_method(method='spawn')

    queue_img_l = [mp.Queue(maxsize=2) for _ in G.camera_ip_l]

    processes = []
    processes.extend([mp.Process(target=queue_img_put, args=(q, G.user_name, G.user_pwd, camera_ip))
                      for (q, camera_ip) in zip(queue_img_l, G.camera_ip_l)])
    processes.extend([mp.Process(target=queue_img_get, args=(q, camera_ip))
                      for (q, camera_ip) in zip(queue_img_l, G.camera_ip_l)])

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
pass
