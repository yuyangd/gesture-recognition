"""
Serve webcam images from a Redis store using Tornado.
Usage:
   python server.py
"""

import base64
import re
from io import BytesIO
from PIL import Image
import time

import numpy as np
from tornado import websocket, web, ioloop
import torch
import torchvision

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

MAX_FPS = 100

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('index-video.html')

class SocketHandler(websocket.WebSocketHandler):
    """ Handler for websocket queries. """
    
    def __init__(self, *args, **kwargs):
        """ Initialize the Redis store and framerate monitor. """

        super(SocketHandler, self).__init__(*args, **kwargs)
        self.load_model()

    def load_model(self):
        self.model = torchvision.models.resnet18()

        self.model.fc = torch.nn.Linear(512, 3)

        self.model.load_state_dict(torch.load('my_model_v2.pth', map_location=torch.device('cpu')))
        self.model.eval()


    def getImgFromBase64(self, codec):
        try:
            base64_data = re.sub('^data:image/.+;base64,', '', codec)
            byte_data = base64.b64decode(base64_data)
            image_data = BytesIO(byte_data)
            image = Image.open(image_data)
            return image
        except:
            return None

    def getMove(self, index):
        moves = ["cowboy","ninja", "bear"]
        return moves[index]

    def getImgModel(self, img_tensor):
        try:
            return self.model(img_tensor)
        except:
            return None

    def on_message(self, message):
        image = self.getImgFromBase64(message)
        if image:
            image_tensor = ToTensor()(image).unsqueeze(0)
            
            img_model = self.getImgModel(image_tensor)
            if img_model is not None:
                output = F.softmax(img_model, dim=1).detach().cpu().numpy().flatten()
                print(output)
                category_index = output.argmax()

                response = {
                    "move": self.getMove(category_index)
                }
            else:
                response = {
                    "move": "nothing"
                }
        else:
            response = {
                "move": "nothing"
            }

        self.write_message(response)

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
])

if __name__ == '__main__':
    app.listen(9000)
    ioloop.IOLoop.instance().start()
