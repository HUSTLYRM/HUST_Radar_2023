import time
import my_serial as messager


last = time.time()
portx = 'COM3'
ser = messager.serial_init(portx)

while True:
    now = time.time()
    # 距离上一次发送时间小于0.1s:sleep
    if now - last < 0.1:
        time.sleep(0.1 - (now - last))
    messager.send_enemy_location(ser, 101, 10., 10.)  # mm to m
    print('send')
    last = time.time()
