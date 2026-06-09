import io
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
device = torch.device("cpu")

# 类别顺序必须和你训练时完全一致
CLASSES = ["transverse","oblique","spiral","comminuted","impacted",
           "avulsion","greenstick","stress","pathologic","normal"]

# ---- 加载模型（按你实际的搭建方式改）----
import timm
model = timm.create_model("efficientnet_b0", num_classes=len(CLASSES))
model.load_state_dict(torch.load("model/efficientnet_b0.pth", map_location=device))
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ---- Grad-CAM 钩子，挂在 conv_head 上 ----
_act, _grad = {}, {}
target_layer = model.conv_head          # timm 的 EfficientNet 最后一层卷积
target_layer.register_forward_hook(lambda m,i,o: _act.update(v=o.detach()))
target_layer.register_full_backward_hook(lambda m,gi,go: _grad.update(v=go[0].detach()))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    W, H = img.size

    x = tf(img).unsqueeze(0).to(device)
    x.requires_grad_(True)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())

    # 反向传播求 Grad-CAM
    model.zero_grad()
    logits[0, idx].backward()
    act = _act["v"][0]                      # C,h,w
    weights = _grad["v"][0].mean(dim=(1,2)) # C
    cam = F.relu((weights[:,None,None]*act).sum(0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (W, H))

    # 热区阈值 -> 骨折红框（百分比坐标）
    mask = (cam > 0.5).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs):
        x0,x1,y0,y1 = xs.min(),xs.max(),ys.min(),ys.max()
    else:
        x0,y0,x1,y1 = 0,0,W,H
    bbox = {"x":x0/W*100, "y":y0/H*100, "w":(x1-x0)/W*100, "h":(y1-y0)/H*100}

    top = torch.topk(probs, 4)
    out_probs = [[CLASSES[i], round(float(probs[i])*100,1)] for i in top.indices.tolist()]

    return jsonify({
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx])*100, 1),
        "probs": out_probs,
        "bbox": bbox
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)