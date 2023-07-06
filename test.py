import os


def save_data(file_path, title, data):
    with open(file_path, 'a') as file:
        file.write(f"{title}: {data}\n")


SavePath = os.getcwd()
RtName = SavePath + 'm_r.txt'

for i in range(1, 11):
    file_path = "data.txt"
    title = "Temperature"
    data = 25.5
    save_data(file_path, title, data)





