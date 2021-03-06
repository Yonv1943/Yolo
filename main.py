from test import *

import multiprocessing as mp
import cv2


def queue_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    is_opened, frame = cap.read()

    '''loop'''
    while is_opened:
        is_opened, frame = cap.read()
        q.put([is_opened, frame])
        if q.qsize() > 2:
            q.get()


def queue_img_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    is_opened = True

    '''TensorFlow init'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    print(os.path.exists(weight_file), weight_file)
    detector = Detector(yolo, weight_file)

    '''loop'''
    detect_timer = Timer()
    while is_opened:
        (is_opened, image) = q.get()

        detect_timer.tic()
        result = detector.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        detector.draw_result(image, result)

        cv2.imshow(window_name, image)
        cv2.waitKey(1)


def main():
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"

    camera_ip_l = [
        # "192.168.1.164",
        # "192.168.1.165",
        # "192.168.1.166",
        "192.168.1.170",
    ]

    mp.set_start_method(method='spawn')

    '''queue'''
    queue_img_l = [mp.Queue(maxsize=64) for _ in camera_ip_l]

    '''process'''
    process_io_2dl = [
        [mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)),
         mp.Process(target=queue_img_get, args=(queue, camera_ip))]
        for (queue, camera_ip) in zip(queue_img_l, camera_ip_l)
    ]

    '''start'''
    for process_l in process_io_2dl:
        for process in process_l:
            process.start()

    '''join'''
    for process_l in process_io_2dl:
        for process in process_l:
            process.join()


if __name__ == '__main__':
    main()
pass
