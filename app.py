import torch
import requests
import io
import PIL.Image, PIL.ImageDraw
import numpy as np
import torch.nn.functional as F
import pygame

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True

def imread(url, max_size=None, mode=None):
  if url.startswith(('http:', 'https:')):
    # wikimedia requires a user agent
    headers = {
      "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
    }
    r = requests.get(url, headers=headers)
    f = io.BytesIO(r.content)
  else:
    f = url
  img = PIL.Image.open(f)
  if max_size is not None:
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  if mode is not None:
    img = img.convert(mode)
  img = np.float32(img)/255.0
  return img

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def to_rgb(x):
  rgb, a = x[:,:3], x[:,3:4]
  return 1.0-a+rgb

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

P = 12  #@param {type:"integer"}
W = 48  #@param {type:"integer"}
WP = W + 2*P
CHN = 16
LAP = torch.tensor([[-1.0, -1.0, -1.0],
                    [-1.0, 8.0, -1.0],
                    [-1.0, -1.0, -1.0]])

emoji = '🦎🕺🏻❤️😁'[0]
code = hex(ord(emoji))[2:].lower()
url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
target = imread(url, W)
target[:,:,:3] *= target[:,:,3:]
target = F.pad(torch.tensor(target).permute(2, 0, 1), [P, P, P, P, 0, 12])
xy = torch.where(target[3]>0.0)
green_target = torch.clone(target[None, ...])
red_target = torch.clone(target[None, ...])
blue_target = torch.clone(target[None, ...])
for x, y in zip(xy[0], xy[1]):
  red_target[0, :3, x, y] = torch.roll(red_target[0, :3, x, y], -1)
  blue_target[0, :3, x, y] = torch.roll(blue_target[0, :3, x, y], 1)
imgs = torch.cat([green_target[:, :4].permute(0, 2, 3, 1).cpu(), 
                  red_target[:, :4].permute(0, 2, 3, 1).cpu(), 
                  blue_target[:, :4].permute(0, 2, 3, 1).cpu()], 0)
# imshow(zoom(tile2d(imgs, 3), 4))

def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = F.pad(y, [1, 1, 1, 1], 'circular')
  y = F.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)

def perception(state):
  lap_perc = perchannel_conv(state, LAP[None, :])
  return torch.cat([state, lap_perc], 1)

class CA(torch.nn.Module):
  def __init__(self, chn=CHN, hidden_n=128):
    super().__init__()
    self.chn = chn
    # determene the number of perceived channels
    perc_n = perception(torch.zeros([1, chn, 8, 8])).shape[1]
    # approximately equalize the param number btw model variants
    hidden_n = 8*1024//(perc_n+chn)
    hidden_n = (hidden_n+31)//32*32
    # print('perc_n:', perc_n, 'hidden_n:', hidden_n)

    self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()

  def forward(self, x, update_rate=0.5):
    alpha = F.pad(x[:,3:4], [1, 1, 1, 1], 'circular')
    alive = F.max_pool2d(alpha, 3, 1)>0.1
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    x = x + y*update_mask
    x = x*alive
    return x

class Viewer:
    def __init__(self, update_f, sz=(WP*6, WP*6)):
        self.update_f = update_f
        self.sz = sz
        pygame.init()
        self.display = pygame.display.set_mode(sz)
        self.x = torch.clone(green_target)

    def start(self):
        running = True
        with torch.no_grad():
            start_time = pygame.time.get_ticks()
            wait_time = 100
            i = 0
            while running:
                # self.display.fill((0, 0, 0))
                current_time = pygame.time.get_ticks()
                if current_time - start_time < wait_time:
                    start_time = current_time
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.x[:, 4:] = 0.0
                            self.x[:, -1] = 1.0
                        if event.key == pygame.K_r:
                            self.x = torch.clone(green_target)
                Z, self.x = self.update_f(self.x, i)
                bytes = Z.flatten()
                surf = pygame.image.frombuffer(bytes, self.sz, 'RGBA')
                # surf = pygame.surfarray.make_surface(Z)
                self.display.blit(surf, (0, 0))
                pygame.display.flip()
                i += 1
        pygame.quit()

# model = torch.load('animation_nca_5000.pt')
model = torch.load('3_animation_nca_5000.pt')

def update(x, i):
    # img = to_rgb(green_target)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    img = zoom(x[0, :4].permute(1, 2, 0).cpu().numpy(), 6)
    img[img[:, :, -1] > 0.1] *= 255.0
    if i % 2 == 0:
        x = model(x)
    return img.astype('uint8'), x

viewer = Viewer(update)
viewer.start()
