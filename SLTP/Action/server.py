import socket
import struct
import sys
import time
import os

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('',5172)) #5172 => sltp
server.listen(5)

while True:
    client, address = server.accept()
    print("connected", address)
    data = ""
                     
    fileSize = client.recv(8) 
    #print(fileSize) #: b'\x00\x00\x......'
    #size = int.from_bytes(fileSize, "big") #python 3.x
    size = struct.unpack(">q", fileSize)[0] #python 2.x   
       
    img_file = open("/home/pi/tf_pi/Translating/inputdata/socket.mp4", "wb")
        
    while True:
        img_data = client.recv(1024)
        data += img_data #python 2.x
            
        if (sys.getsizeof(data)>=size):
            break
        if not img_data:
            break
            #print("receiving Images")
        
    print(size)
    
    connect_file = open("/home/pi/tf_pi/connect.txt","w")
    
    
    img_file.write(data)
    img_file.close()

    text_file = "/home/pi/tf_pi/Translated/result.txt"
    while True:
        if (os.path.isfile(text_file) and os.path.getsize(text_file)>0):
            result_file = open(text_file,"r")
            
            string = result_file.read()
            print(string)
            result_file.close()
            
            os.remove(text_file)
            break
                
    client.send(string.decode('utf-8').encode('euc-kr'))
    client.close()