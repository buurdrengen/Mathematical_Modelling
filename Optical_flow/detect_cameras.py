"""
Test the ports and returns a tuple with the available ports and the ones that are working.
Copied from https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
 at 18:19 09/02/2023


"""
import cv2


def list_ports():

    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []

    print(20*"-")

    for i in range(10):
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1

    
    print(20*"-")
    return available_ports,working_ports


if __name__ == "__main__":
    list_ports()