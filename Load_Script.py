import requests
import random
import threading



def load1():
    threading.Timer(0,load1).start()
    try:
        result = requests.get(url="http://localhost:8080/message")
        print(result.text)
    except:
        pass
    try:
        result2 = requests.get(url="http://localhost:8081/home")
        print(result2.text)
    except:
        pass
    try:
        result3 = requests.get(url="http://localhost:8082/Homepage")
        print(result3.text)
    except:
        pass





# def load2():
#     print(time_var)
#     threading.Timer(0,load2).start()
#     result2 = requests.get(url="http://localhost:8081/home")
#     print(result2.text)
#
#
# def load3():
#     print(time_var)
#     threading.Timer(0,load3).start()
#     result3 = requests.get(url="http://localhost:8082/Homepage")
#     print(result3.text)


# prepare_data()
load1()
# load2()
# load3()