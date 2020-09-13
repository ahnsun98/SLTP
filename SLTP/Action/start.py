import os

videoFile = "/home/pi/tf_pi/Translating/inputdata/socket.mp4"
predict = "python3 predict.py --input_data_path='/home/pi/tf_pi/Translating/' --output_data_path='/home/pi/tf_pi/Translated/'"
checkConnect = "/home/pi/tf_pi/connect.txt"

while True:
    if (os.path.isfile(videoFile) and os.path.isfile("/home/pi/tf_pi/connect.txt")):
        print('start predict!')
        if(os.system(predict) != 0):
            print('predict error')
            break
        print('remove video')
        os.remove(videoFile)
        os.remove(checkConnect)
    else:
        continue