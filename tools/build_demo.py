"""把实际仿真录像和证据图表编排成小于5分钟的字幕演示，不生成虚构运动。"""
from pathlib import Path
import cv2
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
dest=root/'docs/demo';dest.mkdir(parents=True,exist_ok=True)
path=dest/'final_demo.mp4'
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,640))
assert writer.isOpened()
frames=0

def card(title,lines,seconds,picture=None):
    global frames
    frame=np.full((640,960,3),(30,24,20),np.uint8)
    cv2.putText(frame,title,(35,60),cv2.FONT_HERSHEY_SIMPLEX,.9,(60,220,240),2,cv2.LINE_AA)
    if picture:
        image=cv2.imread(str(picture));assert image is not None
        ratio=min(890/image.shape[1],440/image.shape[0]);image=cv2.resize(image,None,fx=ratio,fy=ratio)
        x=(960-image.shape[1])//2;frame[100:100+image.shape[0],x:x+image.shape[1]]=image
        y=570
    else:y=125
    for text in lines:
        cv2.putText(frame,text,(35,y),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA);y+=43
    for _ in range(int(seconds*10)):writer.write(frame);frames+=1

def clip(filename,label):
    global frames
    source=root/'results/demo-visible/videos'/filename
    cap=cv2.VideoCapture(str(source));assert cap.isOpened(),source
    while True:
        ok,frame=cap.read()
        if not ok:break
        cv2.rectangle(frame,(0,0),(960,35),(15,15,15),-1)
        cv2.putText(frame,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,240,240),1,cv2.LINE_AA)
        writer.write(frame);frames+=1
    cap.release()
    return {'file':filename,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}

card('TurtleBot3 Navigation and Visual Docking',[
    'Track A | MuJoCo | RGB camera, encoders and LiDAR',
    'Navigate a static room and dock at the ArUco station.',
    'The controller uses estimated state throughout.',
    'All motion shown below is recorded from actual simulation.',
    'Captioned demo; team names and group number belong on the project page.'
],15)
card('Sensing, estimation and control',[
    'Encoder increments + calibrated ArUco pose -> EKF',
    'LiDAR + estimated pose -> occupancy map -> A star path',
    'Moving-reference LQR -> scan clearance gate -> encoder PI',
    'Marker acquisition -> visual pose docking -> stable stop',
    'True state is shown only for evaluation, not feedback.'
],20)
card('Model and design evidence',[
    'State: x, y, heading. Inputs: linear and angular velocity.',
    'Moving linearisation: controllability rank 3.',
    'Calibrated pose-camera measurement: observability rank 3.',
    'LQR Q = diag(6,40,10), R = diag(4,2).',
    'Closed-loop eigenvalues: 0.9565, 0.9952, 0.9758.',
    'The stationary linearisation has rank 2; docking uses nonlinear feedback.'
],22)
card('1. Nominal docking',[
    'Watch the blue charging area and the actual camera image.',
    'The lower panel compares estimate, truth and marginal +/-2 sigma.'
],8)
sources=[clip('nominal__demo4-000.mp4','NOMINAL | real-time simulation footage | seed 0')]
card('2. Obstacle navigation and docking',[
    'The obstacle position is not supplied to the planner.',
    'The map comes from LiDAR rays transformed by estimated pose.',
    'The robot may reacquire the marker and back away to realign.'
],12)
sources.append(clip('obstacle__demo5-002.mp4','OBSTACLE | actual LiDAR navigation and ArUco docking | seed 2'))
card('3. Temporary LiDAR loss',[
    'All LiDAR returns are missing from simulation time 3 to 5 seconds.',
    'The robot stops, then continues after valid data returns.',
    'This case includes the obstacle and final visual docking.'
],12)
sources.append(clip('lidar_dropout__demo6-002.mp4','LIDAR DROPOUT 3-5 s | actual disturbed trial | seed 2'))
card('Repeated trials: six conditions, three seeds each',[
    '18 completed trials passed; no collisions.',
],20,root/'docs/results/navigation/report/success_rate.png')
card('Final position errors',[
    'Success threshold: true position <=5 cm and heading <=5 degrees.'
],20,root/'docs/results/navigation/report/final_position_error.png')
card('Calibration and limitations',[
    '12 known-pose camera checks: position RMSE 0.00576 m.',
    'Virtual intrinsics from model configuration; no real sensor claim.',
    'One static obstacle, fixed landmark, small start/layout changes.',
    'Reacquisition/reapproach can be slow; not time-optimal.',
    'One execution was interrupted; the same seed passed on restart.',
    '18 completed trials do not guarantee arbitrary layouts or hardware.'
],22)
card('Reproduce and review',[
    'python run_navigation.py --gui --speed 2',
    'python run_navigation.py --condition obstacle --trials 3',
    'Raw states, covariance, sensor logs and configurations are supplied.',
    'AI-assisted implementation, tests, documentation and demo generation.',
    'Team review, attribution, publication and individual viva remain required.'
],15)
writer.release()
assert frames/10<300
(dest/'provenance.json').write_text(json.dumps({'duration_s':frames/10,'fps':10,'size':[960,640],
    'sources':sources,'processing':'Recorded simulation clips and static evidence cards; no generated motion.',
    'audio':'No narration; English captions'},indent=2),encoding='utf-8')
print({'duration_s':frames/10,'size_bytes':path.stat().st_size})
