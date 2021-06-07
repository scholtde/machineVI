#!/usr/bin/env python3

"""
Copyright (c) 2020, Rhizoo Christos Technologies. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import time
import random
import os
import time
import pygame, sys
import pygame.locals
# import RPi.GPIO as GPIO
from uuid import uuid1
# from jetbot import ObjectDetector
from packages import Camera
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
# from jetbot import bgr8_to_jpeg

from flask import Response
from flask import Flask
from flask import render_template
import threading


# Data collector
print("Setup DATA Collector")
normal_dir = 'dataset/normal'
damage_dir = 'dataset/damage'
try:
    os.makedirs(normal_dir)
    os.makedirs(damage_dir)
except FileExistsError:
    print('Directories not created because they already exist')
normal_count = len(os.listdir(normal_dir))
damage_count = len(os.listdir(damage_dir))
print("Normal Count : ", normal_count)
print("Damage Count : ", damage_count)

# Camera
width = 300
height = 300
camera = Camera(width=width, height=height, rotate=False)

# # SSD Object detector
# print("Loading SSD Object Detector")
# model = ObjectDetector('models/object_detect/ssd_mobilenet_v2_coco.engine')

# Collision Detector
print("Loading Collision Model")
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('models/classification/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

# Timers
check_time_damage = time.time()
check_time_normal = time.time()

# Web UI (flask)
outputFrame = None
lock = threading.Lock()  # lock used to ensure thread-safe (multi browsers/tabs)
app = Flask(__name__)
# app.run(host="0.0.0.0", port="8000", debug=True, threaded=True, use_reloader=False)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Functions
def train_bot():
    # Create dataset instance
    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # Split dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])

    # Create data loaders to load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    # Define the neural network
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    t_model = models.alexnet(pretrained=True)  # Download if not exist
    t_model.classifier[6] = torch.nn.Linear(t_model.classifier[6].in_features, 2)

    compute_device = torch.device('cuda')
    t_model = t_model.to(compute_device)

    # Train the neural network
    # noinspection PyPep8Naming
    NUM_EPOCHS = 50
    # noinspection PyPep8Naming
    BEST_MODEL_PATH = 'models/classification/best_model.pth'
    best_accuracy = 0.0

    optimizer = optim.SGD(t_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        for images, labels in iter(train_loader):
            images = images.to(compute_device)
            labels = labels.to(compute_device)
            optimizer.zero_grad()
            outputs = t_model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(compute_device)
            labels = labels.to(compute_device)
            outputs = t_model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))

        # Write to UI
        t_y = 110
        t_x = 200
        size = 30
        w = 250
        offset = 43
        rec = 0, 77, 77
        txtCol = 255, 255, 255

        write_text(1, str(normal_count), 5, t_x, t_y, size, w, size, txtCol, rec)
        t_y = t_y + offset
        write_text(1, str(damage_count), 5, t_x, t_y, size, w, size, txtCol, rec)
        t_y = t_y + offset
        write_text(1, '%d: %f' % (epoch, test_accuracy), 5, t_x, t_y, size, w, size, txtCol, rec)

        if test_accuracy > best_accuracy:
            torch.save(t_model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy

        return True


def prepare_ui(full_screen):
    pygame.display.set_caption('machineVI v.1.0')
    if full_screen:
        s_width = 1024
        s_height = 750
        surface = pygame.display.set_mode((s_width, s_height), pygame.FULLSCREEN, 32)
    else:
        s_width = 510
        s_height = 680
        surface = pygame.display.set_mode((s_width, s_height), 0, 32)
 
    icon = pygame.image.load('img/RCT_Logo_205x71l.png')
    background = (0, 0, 0)
    x = 90  # (WIDTH * 0.90)
    y = 12  # (HEIGHT * 0.80)
    surface.fill(background)
    surface.blit(icon, (x, y))
    pygame.display.set_icon(icon)
    pygame.display.update()
    pygame.display.flip()

    return surface

         
def write_text(path, text, xoffset, x, y, size, w, h, text_col, rect_col):
    switcher = {
      0: "fonts/LCARS.ttf",
      1: "fonts/LCARSGTJ2.ttf",
      2: "fonts/LCARSGTJ3.ttf",
    }
    # windowSurface.fill(background)
    # windowSurface.blit(Image, (x,y))
    rectangle = pygame.Rect(x-5, y, w, h*1.1)
    pygame.draw.rect(windowSurface, rect_col, rectangle, 0)
    font = pygame.font.Font(switcher.get(path, "fonts/LCARSGTJ2.ttf"), size)
    label = font.render(text, 1, text_col)
    windowSurface.blit(label, (x+xoffset, y))
    # pygame.display.flip()
    pygame.display.update(rectangle)

    return rectangle


def ui_labels():
    # (path, text, xoffset, x, y, size, w, h, text_col, recCol)
    y = 12
    x = 300
    offset = 0
    rec = 0, 0, 0
    write_text(1, "machineVI", offset, x, y, 50, 150, 50, (255, 255, 255), rec)
    y = 55
    write_text(1, "Version v1.0", offset, x, y, 20, 150, 20, (29, 136, 133), rec)

    y = 110
    x = 19
    size = 30
    w = 170
    offset = 43
    rec = 0, 77, 77
    txt_col = 255, 255, 255

    write_text(1, "NOR-IMG", 95, x, y, size, w, size, txt_col, rec)
    y = y+offset
    write_text(1, "DAM-IMG", 95, x, y, size, w, size, txt_col, rec)
    y = y+offset
    write_text(1, "STATUS", 100, x, y, size, w, size, txt_col, rec)

    # Default values
    y = 110
    x = 200
    w = 250
    offset = 43

    write_text(1, str(normal_count), 5, x, y, size, w, size, txt_col, rec)
    y = y + offset
    write_text(1, str(damage_count), 5, x, y, size, w, size, txt_col, rec)
    y = y + offset
    write_text(1, "DONE", 5, x, y, size, w, size, txt_col, rec)
    # --------------
 
    x = 12
    y = 232
    size = 55
    w = 200
    offset = 60
    rec = 0, 0, 0
    txt_col = 255, 255, 255
    write_text(1, "MODE:", 80, x, y, size, w, size, txt_col, rec)
    y = y+offset
    write_text(1, "COND:", 85, x, y, size, w, size, txt_col, rec)

    x = 12
    y = y+offset
    # windowSurface.blit(image, (x, y))
    rectangle = pygame.Rect(x, y, 480, 270)
    pygame.draw.rect(windowSurface, (50, 50, 50), rectangle, 0)
    pygame.display.update(rectangle)


def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    # save image
    image = camera.image_array
    image = cv2.resize(image, (224, 224))
    status = cv2.imwrite(image_path, image)
    print("Image written to file-system : ", status)
    # with open(image_path, 'wb') as f:
    #    f.write(image.image_array)


def save_normal():
    global normal_dir, normal_count
    save_snapshot(normal_dir)
    normal_count = len(os.listdir(normal_dir))
    print("Normal Count : ", normal_count)
    print("Damage Count : ", damage_count)

    
def save_damage():
    global damage_dir, damage_count
    save_snapshot(damage_dir)
    damage_count = len(os.listdir(damage_dir))
    print("Normal Count : ", normal_count)
    print("Damage Count : ", damage_count)


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return center_x, center_y


def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def pause(c_time, delay):
    if time.time() - c_time > delay:
        return True
    else:
        return False

 
def execute(change):
    global check_time_damage
    global check_time_normal
    global outputFrame, lock

    image = change['new']

    # execute collision model to determine the condition
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_cond = float(F.softmax(collision_output.flatten(), dim=0)[0])
    # blocked_widget.value = prob_blocked

    mode = "INFER"
    # turn right if blocked
    if prob_cond >= 0.50:
        path = str(round(prob_cond * 100, 1)) + "% - DAMAGE"
        if pause(check_time_damage, 0.7):
            print("DAMAGE detected")
            check_time_damage = time.time()

    # If robot is not blocked, move towards target
    else:
        path = str(round(prob_cond * 100, 1)) + "% - NORMAL"
        check_time_normal = time.time()

    x = 200
    y = 232
    size = 55
    w = 270
    offset = 60
    rec = 0, 0, 0
    txt_col = 255, 255, 255
    write_text(1, mode, 5, x, y, size, w, size, txt_col, rec)
    y = y+offset
    write_text(1, path, 5, x, y, size, w, size, txt_col, rec)

    # Update image
    # image_widget.value = bgr8_to_jpeg(image)
    image = cv2.resize(image, (480, 270))
    # acquire the lock, set the output frame, and release the
    # lock
    with lock:
        outputFrame = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.swapaxes(0, 1)
    image = pygame.surfarray.make_surface(image)
    x = 12
    y = y+offset
    windowSurface.blit(image, (x, y))
    rectangle = pygame.Rect(x, y, 480, 270)
    pygame.display.update(rectangle)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


def main():
    while True:
        # Check Ctrl+C
        try:
            # detections = model(camera.value)
            # print(detections)
            execute({'new': camera.image_array})
           
        except KeyboardInterrupt:
            camera.stop()
            pygame.quit()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    camera.stop()
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_r:
                    camera.exec_rotate()

                if event.key == pygame.K_n:
                    time.sleep(1)
                    print("Save NORMAL")
                    save_normal()

                    t_y = 110
                    t_x = 200
                    size = 30
                    w = 250
                    offset = 43
                    rec = 0, 77, 77
                    txtCol = 255, 255, 255

                    btnAL = write_text(1, str(normal_count), 5, t_x, t_y, size, w, size, txtCol, rec)
                    t_y = t_y + offset
                    btnVT = write_text(1, str(damage_count), 5, t_x, t_y, size, w, size, txtCol, rec)
                    t_y = t_y + offset
                    btnVT = write_text(1, "DONE", 5, t_x, t_y, size, w, size, txtCol, rec)


                if event.key == pygame.K_d:
                    time.sleep(1)
                    print("Save DAMAGE")
                    save_damage()

                    t_y = 110
                    t_x = 200
                    size = 30
                    w = 250
                    offset = 43
                    rec = 0, 77, 77
                    txtCol = 255, 255, 255

                    btnAL = write_text(1, str(normal_count), 5, t_x, t_y, size, w, size, txtCol, rec)
                    t_y = t_y + offset
                    btnVT = write_text(1, str(damage_count), 5, t_x, t_y, size, w, size, txtCol, rec)
                    t_y = t_y + offset
                    btnVT = write_text(1, "DONE", 5, t_x, t_y, size, w, size, txtCol, rec)

                if event.key == pygame.K_t:
                    time.sleep(1)
                    print("Training mode..")
                    x = 200
                    y = 232
                    size = 55
                    w = 270
                    offset = 60
                    rec = 0, 0, 0
                    txt_col = 255, 255, 255
                    write_text(1, "TRAIN", 5, x, y, size, w, size, txt_col, rec)
                    y = y + offset
                    write_text(1, "WAIT..", 5, x, y, size, w, size, txt_col, rec)

                    if train_bot():
                        # Reload Collision Detector
                        print("Loading Collision Model")
                        collision_model = torchvision.models.alexnet(pretrained=False)
                        collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
                        collision_model.load_state_dict(torch.load('models/classification/best_model.pth'))
                        device = torch.device('cuda')
                        collision_model = collision_model.to(device)
                        mean = 255.0 * np.array([0.485, 0.456, 0.406])
                        stdev = 255.0 * np.array([0.229, 0.224, 0.225])
                        normalize = torchvision.transforms.Normalize(mean, stdev)

                        print("Training Complete!")


# Load UI
print("Loading UI")
pygame.init()
windowSurface = prepare_ui(0)

# Execute MAIN
if __name__ == '__main__':
    ui_labels()
    main()
