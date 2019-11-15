"""
Serve webcam and process image through ML model.
Usage:
   python server.py
"""

import base64
import re
from io import BytesIO
from PIL import Image

import numpy as np
from tornado import websocket, web, ioloop
import torch
import torchvision

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F

mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
std = torch.Tensor([0.229, 0.224, 0.225]).cpu()

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('index-video.html')

class GameHandler(web.RequestHandler):
    def get(self):
        self.render('index-game.html')


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
            img_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
            return self.model(img_tensor[None, ...])
            #return self.model(img_tensor)
        except:
            return None

    def on_message(self, message):
        image = self.getImgFromBase64(message)
        if image is not None:
            image_tensor = transforms.functional.to_tensor(image).to(torch.device('cpu'))
            img_model = self.getImgModel(image_tensor)
            if img_model is not None:
                # print("image model")
                # torch.no_grad()
                output = F.softmax(img_model, dim=1).detach().cpu().numpy().flatten() # TODO dim=0?
                print(output)
                category_index = output.argmax()

                response = {
                    "move": self.getMove(category_index),
                    "image": message
                }
            else:
                print("Image model is none")
                response = {
                    "move": "nothing"
                }
        else:
            print("Image is none")
            response = {
                "move": "nothing"
            }

        self.write_message(response)

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
    (r'/game', GameHandler),
])

if __name__ == '__main__':
    app.listen(9000)
    ioloop.IOLoop.instance().start()
