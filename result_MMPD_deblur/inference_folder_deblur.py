from utils import *
from models import * 
from unsupervised_methods import * 
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Video path")
parser.add_argument("--out", help="BVP csv file path", default='')
parser.add_argument("--weights", help="HDF5 path of model weights", default='auto') 
parser.add_argument("--model", help="Supported models: seq, tscan, deepphys, efficientphys, physnet, chrom, pos, ica", default='seq') 
parser.add_argument("--fps", help="Sample the video to the target fps", default='30')  
parser.add_argument("--show-wave", help="Display waveform", action='store_true')

args = parser.parse_args()

if args.model == 'seq':
    resolution = (8, 8) 
    seq = M_1()
    seq.build(input_shape=(None, 450, 8, 8, 3)) 
    if args.weights == 'auto':
        seq.load_weights('./weights/m1.h5')
    else:
        seq.load_weights(args.weights)
    model = lambda x:seq(np.array([x]))[0] 
    chunk = 450 
    cumsum = False
elif args.model == 'tscan':
    resolution = (36, 36) 
    tscan = TS_CAN_end_to_end(n=20) 
    tscan.build(input_shape=(None, 36, 36, 3))
    if args.weights == 'auto':
        tscan.load_weights('./weights/TS-CAN_CCNU.h5')
    else:
        tscan.load_weights(args.weights)
    model = tscan
    chunk = 160  
    cumsum = True
elif args.model == 'deepphys':
    resolution = (36, 36) 
    dp = DeepPhys_end_to_end()
    dp.build(input_shape=(None, 36, 36, 3))
    if args.weights == 'auto':
        dp.load_weights('./weights/DeepPhys_CCNU.h5')
    else:
        dp.load_weights(args.weights) 
    model = dp  
    chunk = None  
    cumsum = True 
elif args.model == 'efficientphys':
    resolution = (72, 72) 
    ep = EP(n=32)
    ep.build(input_shape=(None, 72, 72, 3))
    if args.weights == 'auto':
        ep.load_weights('./weights/EfficientPhys_CCNU.h5')
    else:
        ep.load_weights(args.weights)
    model = ep  
    chunk = 32   
    cumsum = True 
elif args.model == 'physnet':
    resolution = (32, 32)  
    phys_net = PhysNet()
    phys_net.build(input_shape=(None, 128, 32, 32, 3))
    if args.weights == 'auto':
        phys_net.load_weights('./weights/PhysNet_CCNU.h5')  
    else:
        phys_net.load_weights(args.weights)
    model = lambda x:phys_net(np.array([x]))[0]  
    chunk = None 
    cumsum = False  
elif args.model == 'chrom':
    resolution = (1, 1) 
    model = lambda x:CHROM(np.mean(x, axis=(-3, -2))) 
    chunk = None
    cumsum = False  
elif args.model == 'pos': 
    resolution = (1, 1) 
    model = lambda x:POS(np.mean(x, axis=(-3, -2)), fs=float(args.fps)).reshape(-1)
    chunk = None
    cumsum = False  
elif args.model == 'ica': 
    resolution = (1, 1) 
    model = lambda x:ICA(np.mean(x, axis=(-3, -2))) 
    chunk = None
    cumsum = False 


def read_images(folder_path):
    # 生成需要读取的文件名列表
    # filenames = [f"{i:06d}.png" for i in range(1, 1800)]  # 生成 "000001.png" 到 "001799.png"
    filenames = [f"{i:06d}.png" for i in range(1, 1800)]  # 生成 "000001.png" 到 "001799.png"
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        # print(file_path)
        if os.path.exists(file_path):  # 检查文件是否存在
            image = cv2.imread(file_path)
            if image is not None:
                yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print(f"Warning: Failed to read image {filename}")
        else:
            print(f"Warning: File not found {filename}")
    
def vid(v):
    cap = cv2.VideoCapture(v)
    while 1:
        _, f = cap.read()
        if not _:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

folder_root = "/home/dejavu/Code/UMSN-Face-Deblurring/mediapipe_walking_deblur"
out_folder_path = "/home/dejavu/Code/PhysBench/deblur_rppg"

folder_path_list = [] 
for folder_name in os.listdir("/home/dejavu/Code/UMSN-Face-Deblurring/mediapipe_walking_deblur"):
    folder_path_list.append(os.path.join(folder_root, folder_name))
fps = 30
bvp = []
f = []
n = 0
for folder_path in folder_path_list:
    bvp = []
    f = []
    n = 0
    v = folder_path.split('/')[-1]
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1) as fm:
        box = box_ = None
        for frame in read_images(folder_path): #去模糊后的图片文件夹
            h, w, c = frame.shape
            if w>h:
                frame_ = frame[:, round((w-h)/2):round((w-h)/2)+h] #Crop the middle part of the widescreen to avoid detecting other people.
            else:
                frame_ = frame
                w = h
            if n%5==0:
                landmarks = fm.process(frame_).multi_face_landmarks
                if landmarks and len(landmarks): #If a face is detected, convert it to relative coordinates; otherwise, set all values to -1.
                    landmark = np.array([(i.x*h/w+round((w-h)/2)/w, i.y) for i in landmarks[0].landmark])
                    shape = alphashape.alphashape(landmark, 0)
                    if box is None:
                        box = np.array(shape.bounds).reshape(2, 2)
                    else:
                        w = 1/(1 + np.exp(-20*np.linalg.norm(np.array(shape.bounds).reshape(2, 2)-box)/np.multiply(*np.abs(box[0]-box[1]))))*2-1
                        box = np.array(shape.bounds).reshape(2, 2)*w+box*(1-w)
                    if box_ is None:
                        box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                    elif np.linalg.norm(np.round(box*frame.shape[1::-1]).astype(int).T - box_) > frame.size/10**5:
                        box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                else:
                    landmark = np.full((468, 2), -1)
            n += 1
            if box_ is None:
                bvp.append(0)
            else:
                _ = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])], resolution, interpolation=cv2.INTER_AREA)
                f.append(_) 
    frames = np.array(f)/255
    p = frames.reshape(frames.shape[0], -1, frames.shape[-1])
    length = int(frames.shape[0] * float(args.fps)/fps) 
    frames = cv2.resize(p, (p.shape[1], length)).reshape(length, *frames.shape[1:]) 
    if chunk:
        n = 0 
        opts = []
        while 1:
            opt = np.full((frames.shape[0],), np.nan)
            ipt = frames[n:n+chunk]
            if ipt.shape[0] < chunk:
                ipt_ = np.concatenate([ipt, [ipt[-1]]*(chunk-ipt.shape[0])], axis=0) 
                opt[n:] = np.array(model(ipt_)).reshape((-1, ))[:ipt.shape[0]] 
                opts.append(opt) 
                break
            opt[n:n+chunk] = np.array(model(ipt)).reshape((-1, )) 
            opts.append(opt) 
            n += chunk // 2
        predict = np.nanmean(np.array(opts), axis=0)  
        if cumsum:
            predict = np.cumsum(predict) 
    else:
        predict = model(frames) 

    predict[np.isnan(predict)] = 0.
    predict = UnivariateSpline(np.linspace(0, 1, predict.shape[0]), predict, s=0)(np.linspace(0, 1, len(f))) 
    predict = bandpass_filter(predict, fs=fps)    
    bvp = np.concatenate([bvp, predict]) 
    bvp = (bvp-bvp.mean())/(bvp.std()+1e-6) 
    bvp = np.clip(bvp, a_max=bvp.std()*3, a_min=-bvp.std()*3)
    out = args.out
    if not out:
        out = os.path.join(out_folder_path, v+'.csv')

    with open(out, 'w') as f:
        f.write('Timestamp, BVP\n')
        for i, j in enumerate(bvp):
            f.write(f'{i/fps},{j}\n')

    print(f'\nHeart Rate: {get_hr(bvp, sr=float(args.fps)):.2f}')  
    print(f'\nBVP output save path: {out}') 

    if args.show_wave:
        plt.plot(*pd.read_csv(out).values.T) 
        plt.show()