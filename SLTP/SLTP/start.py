import os

videoFile = "/home/pi/tf_pi/Translating/inputdata/socket.mp4"
predict = "python3 predict.py --input_data_path='/home/pi/tf_pi/Translating/' --output_data_path='/home/pi/tf_pi/Translated/'"

while True:
    if os.path.isfile(videoFile):
        print('start predict!')
        if(os.system(predict) != 0):
            print('predict error')
            break
        print('remove video')
        os.remove(videoFile)
    else:
        continue