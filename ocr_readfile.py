from aip import AipOcr

APP_ID = '23579406'
API_KEY = 'zj5MgHG4pUaHGYulGKwarxK6'
SECRET_KEY = 'qdvenK8wC2HOkdGQ0jTRvQ71v3GIW4tv'


class MyOcr:
    def __init__(self, APP_ID, API_KEY, SECRET_KEY, path):
        self.client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
        self.path = path
        self.text = ''
        self.read_text_from_path()

    def read_text_from_path(self):
        with open(self.path, 'rb') as f:
            image = f.read()
        dic_result = self.client.basicGeneral(image)
        res = dic_result["words_result"]
        for i in res:
            self.text += i['words']

    def get_text(self):
        return self.text


myocr = MyOcr(APP_ID, API_KEY, SECRET_KEY, '7.jpg')
print(myocr.get_text())