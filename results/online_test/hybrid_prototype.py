import sys
sys.path.insert(0,'/home/smeer/Downloads/MLOE---Group-4/src')
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
import stress_test as st, depth as dav2
from eval_synth import load
from prep_uieb import resize_pair
from day3_touchup import load_touchup
d,npz,out=sys.argv[1:4]
keep=[0,2,3,4,5,6,8,9,10,11,12,14,17,18,19,20,21,23,24,25]
br=np.load(npz); base=load("baseline_v1_60ep_bigdata")[0].requires_grad_(False).eval()
st.TOUCH["day3a"]=load_touchup("day3a_touchup_global")[0]
raws=[]
for i in keep:
    a=np.array(Image.open(f'{d}/{i:02d}.jpg').convert('RGB')); raws.append(resize_pair(a,a)[0])
zs=[np.clip(z,0.1,30) for z in dav2.predict(raws,"metric-indoor",bs=1)]
g=lambda s:(s*[.299,.587,.114]).sum(-1).astype(np.float32)
box=lambda x:cv2.blur(x,(15,15))
def trust(raw,b):
    """Per-pixel weight for BridgeNet: its local structure must match the photo's, on real texture, without blown highlights."""
    x,y=g(raw),g(b); mx,my=box(x),box(y)
    vx=np.maximum(box(x*x)-mx*mx,0); vy=np.maximum(box(y*y)-my*my,0); cov=box(x*y)-mx*my
    corr=cov/np.sqrt(vx*vy+1e-6)
    w=np.clip((corr-.3)/.4,0,1)*np.clip(np.sqrt(vx)/.02,0,1)*(y<.93)
    return cv2.GaussianBlur(w.astype(np.float32),(0,0),6)
W,H=230,170; T=['raw','+ day3a','BridgeNet','trust map (white = BridgeNet)','Hybrid']
half=[Image.new('RGB',(W*5,(H+4)*10+18),'white') for _ in range(2)]
res={}
for r,(i,a,z) in enumerate(zip(keep,raws,zs)):
    I,zt=st.srgb_to_linear(st.t(a/255.0)),st.t(z[...,None].astype(np.float32))
    _,o=st.systems(base,{},I,zt); t3=st.srgb_np(o["TouchUp [day3a]"])
    b=br[f'bridge_{i}']/255.0; w=trust(a/255.0,b)
    h=w[...,None]*b+(1-w[...,None])*t3
    res[f'hybrid_{i}']=(np.clip(h,0,1)*255+.5).astype(np.uint8)
    print(f'#{r+1} mean trust {w.mean():.2f}')
    sh=half[r//10]; dr=ImageDraw.Draw(sh)
    for c,t in enumerate(T): dr.text((c*W+4,3),t,fill='black')
    for c,im in enumerate([a/255.0,t3,b,np.repeat(w[...,None],3,-1),h]):
        p=Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)); p.thumbnail((W-4,H)); sh.paste(p,(c*W+2,18+(r%10)*(H+4)))
    dr.text((2,18+(r%10)*(H+4)),f'#{r+1}',fill='yellow')
half[0].save(out+'_a.png'); half[1].save(out+'_b.png'); np.savez(out+'.npz',**res)
