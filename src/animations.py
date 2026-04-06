"""
APEX Animation Engine v3 — Immersive 3D background system.
Apple Vision Pro · Mercedes AMG · Stripe Atlas quality.
WebGL geometry · Volumetric light · Aurora depth · 3D grid floor.
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests


# ── Lottie ────────────────────────────────────────────────────────────────────
def load_lottie(url: str) -> dict:
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

LOTTIES = {
    "upload": "https://assets9.lottiefiles.com/packages/lf20_urbk83vw.json",
    "robot":  "https://assets4.lottiefiles.com/packages/lf20_myejiggj.json",
    "chart":  "https://assets9.lottiefiles.com/packages/lf20_qtogtiwt.json",
}

def show_lottie(key: str, height: int = 200, col=None):
    anim = load_lottie(LOTTIES.get(key, ""))
    if anim:
        if col:
            with col:
                st_lottie(anim, height=height, key=key, speed=1, loop=True)
        else:
            st_lottie(anim, height=height, key=key, speed=1, loop=True)


def aurora_background() -> str:
    """
    APEX Immersive 3D Background — Apple Vision Pro / Mercedes AMG level.

    Layers (back to front):
    1. Deep void base
    2. Volumetric aurora blobs (animated radial gradients, slow drift)
    3. 3D perspective grid floor (Mercedes AMG / Tron style)
    4. Floating 3D wireframe geometry (rotating torus + icosphere)
    5. Light beam columns (Apple keynote style)
    6. Neural particle mesh (gold nodes + connections)
    7. Floating data orbs with depth parallax
    8. Subtle film grain overlay
    """
    return r"""<!DOCTYPE html><html><head>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
html,body{width:100%;height:100%;overflow:hidden;background:transparent;}
canvas{position:fixed;top:0;left:0;pointer-events:none;display:block;}
#cv-aurora   {width:100vw;height:100vh;z-index:0;}
#cv-grid     {width:100vw;height:100vh;z-index:1;}
#cv-geo      {width:100vw;height:100vh;z-index:2;}
#cv-beams    {width:100vw;height:100vh;z-index:3;}
#cv-particles{width:100vw;height:100vh;z-index:4;}
#cv-orbs     {width:100vw;height:100vh;z-index:5;}
</style></head><body>
<canvas id="cv-aurora"></canvas>
<canvas id="cv-grid"></canvas>
<canvas id="cv-geo"></canvas>
<canvas id="cv-beams"></canvas>
<canvas id="cv-particles"></canvas>
<canvas id="cv-orbs"></canvas>
<script>
'use strict';

/* ═══════════════════════════════════════════════
   SHARED STATE
═══════════════════════════════════════════════ */
let W=window.innerWidth, H=window.innerHeight;
let MX=W/2, MY=H/2;
let T=0;

function resize(){
    W=window.innerWidth; H=window.innerHeight;
    document.querySelectorAll('canvas').forEach(c=>{c.width=W;c.height=H;});
}
window.addEventListener('resize',resize);
document.addEventListener('mousemove',e=>{MX=e.clientX;MY=e.clientY;});
resize();

/* ═══════════════════════════════════════════════
   LAYER 1 — VOLUMETRIC AURORA
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-aurora');
    const cx=cv.getContext('2d');

    const BLOBS=[
        {nx:.5, ny:-.05, rw:.85, rh:.5,  h:42,  s:75, a:.11, sp:.00018, ph:0   },
        {nx:.1, ny:.85,  rw:.55, rh:.38, h:258, s:65, a:.07, sp:.00025, ph:1.57},
        {nx:.9, ny:.75,  rw:.45, rh:.32, h:162, s:60, a:.06, sp:.00030, ph:3.14},
        {nx:.5, ny:.5,   rw:.7,  rh:.5,  h:42,  s:55, a:.04, sp:.00012, ph:4.71},
        {nx:.2, ny:.2,   rw:.4,  rh:.3,  h:200, s:50, a:.04, sp:.00022, ph:2.0 },
    ];

    function draw(t){
        cx.clearRect(0,0,W,H);
        BLOBS.forEach(b=>{
            const ox=Math.sin(t*b.sp*5000+b.ph)*.07;
            const oy=Math.cos(t*b.sp*4000+b.ph)*.05;
            const px=(b.nx+ox)*W, py=(b.ny+oy)*H;
            const rx=Math.max(W,H)*b.rw;
            const a=b.a*(0.65+Math.sin(t*0.4+b.ph)*0.35);
            const g=cx.createRadialGradient(px,py,0,px,py,rx);
            g.addColorStop(0,   `hsla(${b.h},${b.s}%,62%,${a})`);
            g.addColorStop(0.4, `hsla(${b.h},${b.s}%,45%,${a*0.5})`);
            g.addColorStop(1,   'transparent');
            cx.fillStyle=g;
            cx.fillRect(0,0,W,H);
        });
    }

    (function loop(t){requestAnimationFrame(loop);draw(t/1000);})(0);
})();

/* ═══════════════════════════════════════════════
   LAYER 2 — 3D PERSPECTIVE GRID FLOOR
   Mercedes AMG / Tron aesthetic
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-grid');
    const cx=cv.getContext('2d');

    function draw(t){
        cx.clearRect(0,0,W,H);

        const horizon = H*0.62;
        const vp = {x:W/2 + (MX-W/2)*0.04, y:horizon};
        const LINES=28;
        const COLS=22;
        const spread=W*1.8;
        const depth=H*0.55;
        const pulse=0.5+Math.sin(t*0.5)*0.5;

        /* Horizontal lines — receding into distance */
        for(let i=0;i<=LINES;i++){
            const frac=Math.pow(i/LINES,1.8);
            const y=horizon+frac*depth;
            const xLeft =vp.x - spread*(1-frac)*0.5 - spread*frac*0.5;
            const xRight=vp.x + spread*(1-frac)*0.5 + spread*frac*0.5;
            const alpha=(0.03+frac*0.12)*(0.6+pulse*0.4);
            const lw=0.3+frac*0.8;
            cx.beginPath();
            cx.moveTo(xLeft,y); cx.lineTo(xRight,y);
            cx.strokeStyle=`rgba(201,168,76,${alpha})`;
            cx.lineWidth=lw;
            cx.stroke();
        }

        /* Vertical lines — converging to vanishing point */
        for(let i=0;i<=COLS;i++){
            const frac=i/COLS;
            const xBottom=vp.x-spread/2+frac*spread;
            const alpha=(0.02+Math.abs(frac-0.5)*0.1)*(0.5+pulse*0.5);
            cx.beginPath();
            cx.moveTo(vp.x,vp.y);
            cx.lineTo(xBottom,horizon+depth);
            cx.strokeStyle=`rgba(201,168,76,${alpha})`;
            cx.lineWidth=0.4;
            cx.stroke();
        }

        /* Horizon glow line */
        const hg=cx.createLinearGradient(0,horizon,W,horizon);
        hg.addColorStop(0,'transparent');
        hg.addColorStop(0.3,`rgba(201,168,76,${0.15+pulse*0.2})`);
        hg.addColorStop(0.5,`rgba(248,230,140,${0.35+pulse*0.3})`);
        hg.addColorStop(0.7,`rgba(201,168,76,${0.15+pulse*0.2})`);
        hg.addColorStop(1,'transparent');
        cx.beginPath();
        cx.moveTo(0,horizon); cx.lineTo(W,horizon);
        cx.strokeStyle=hg;
        cx.lineWidth=1.5;
        cx.stroke();

        /* Fade mask — top of grid fades to transparent */
        const fade=cx.createLinearGradient(0,horizon-60,0,horizon+80);
        fade.addColorStop(0,'rgba(2,2,8,0.95)');
        fade.addColorStop(1,'transparent');
        cx.fillStyle=fade;
        cx.fillRect(0,horizon-60,W,140);
    }

    (function loop(t){requestAnimationFrame(loop);draw(t/1000);})(0);
})();

/* ═══════════════════════════════════════════════
   LAYER 3 — 3D WIREFRAME GEOMETRY
   Rotating torus + icosphere (Apple Vision Pro style)
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-geo');
    const cx=cv.getContext('2d');

    /* ── 3D math helpers ── */
    function rotX(p,a){
        const c=Math.cos(a),s=Math.sin(a);
        return [p[0], p[1]*c-p[2]*s, p[1]*s+p[2]*c];
    }
    function rotY(p,a){
        const c=Math.cos(a),s=Math.sin(a);
        return [p[0]*c+p[2]*s, p[1], -p[0]*s+p[2]*c];
    }
    function rotZ(p,a){
        const c=Math.cos(a),s=Math.sin(a);
        return [p[0]*c-p[1]*s, p[0]*s+p[1]*c, p[2]];
    }
    function project(p, fov, cx2, cy2){
        const z=p[2]+fov;
        if(z<=0)return null;
        const scale=fov/z;
        return [cx2+p[0]*scale, cy2+p[1]*scale, scale];
    }

    /* ── Build torus ── */
    function buildTorus(R,r,segsR,segsr){
        const verts=[], edges=[];
        for(let i=0;i<segsR;i++){
            const a=i/segsR*Math.PI*2;
            for(let j=0;j<segsr;j++){
                const b=j/segsr*Math.PI*2;
                verts.push([
                    (R+r*Math.cos(b))*Math.cos(a),
                    (R+r*Math.cos(b))*Math.sin(a),
                    r*Math.sin(b)
                ]);
            }
        }
        for(let i=0;i<segsR;i++){
            for(let j=0;j<segsr;j++){
                const a=i*segsr+j;
                const b=i*segsr+(j+1)%segsr;
                const c=((i+1)%segsR)*segsr+j;
                edges.push([a,b],[a,c]);
            }
        }
        return {verts,edges};
    }

    /* ── Build icosphere ── */
    function buildIco(radius, subdivisions){
        const t=(1+Math.sqrt(5))/2;
        let verts=[
            [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
            [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
            [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]
        ].map(v=>{const l=Math.sqrt(v[0]**2+v[1]**2+v[2]**2);return v.map(x=>x/l*radius);});
        let faces=[
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
        ];
        for(let s=0;s<subdivisions;s++){
            const newFaces=[];
            const midCache={};
            function mid(a,b){
                const key=Math.min(a,b)+','+Math.max(a,b);
                if(midCache[key]!==undefined)return midCache[key];
                const va=verts[a],vb=verts[b];
                const m=[(va[0]+vb[0])/2,(va[1]+vb[1])/2,(va[2]+vb[2])/2];
                const l=Math.sqrt(m[0]**2+m[1]**2+m[2]**2);
                verts.push(m.map(x=>x/l*radius));
                return midCache[key]=verts.length-1;
            }
            faces.forEach(([a,b,c])=>{
                const ab=mid(a,b),bc=mid(b,c),ca=mid(c,a);
                newFaces.push([a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]);
            });
            faces=newFaces;
        }
        const edges=new Set();
        faces.forEach(([a,b,c])=>{
            [[a,b],[b,c],[c,a]].forEach(([x,y])=>{
                edges.add(Math.min(x,y)+','+Math.max(x,y));
            });
        });
        return {verts, edges:[...edges].map(e=>e.split(',').map(Number))};
    }

    const TORUS=buildTorus(110,38,28,14);
    const ICO  =buildIco(75,1);

    function drawGeo(geo, cx2, cy2, rx, ry, rz, fov, color, alpha, t){
        const projected=geo.verts.map(v=>{
            let p=rotX(v,rx); p=rotY(p,ry); p=rotZ(p,rz);
            return project(p,fov,cx2,cy2);
        });
        geo.edges.forEach(([a,b])=>{
            const pa=projected[a], pb=projected[b];
            if(!pa||!pb)return;
            const depth=(pa[2]+pb[2])/2;
            const a2=alpha*Math.min(depth*0.8,1)*0.7;
            if(a2<0.005)return;
            cx.beginPath();
            cx.moveTo(pa[0],pa[1]);
            cx.lineTo(pb[0],pb[1]);
            cx.strokeStyle=color.replace('A',a2.toFixed(3));
            cx.lineWidth=0.6;
            cx.stroke();
        });
    }

    function draw(t){
        cx.clearRect(0,0,W,H);
        const mx=(MX/W-0.5)*0.3;
        const my=(MY/H-0.5)*0.2;

        /* Torus — left side, slow rotation */
        drawGeo(TORUS,
            W*0.18, H*0.38,
            t*0.18+my, t*0.28+mx, t*0.08,
            420,
            'rgba(201,168,76,A)', 0.55, t
        );

        /* Icosphere — right side, counter-rotation */
        drawGeo(ICO,
            W*0.84, H*0.32,
            t*0.22+my, -t*0.32+mx, t*0.12,
            380,
            'rgba(201,168,76,A)', 0.45, t
        );

        /* Small torus — center top */
        drawGeo(TORUS,
            W*0.5, H*0.12,
            t*0.12, t*0.20+mx*0.5, 0,
            320,
            'rgba(232,201,106,A)', 0.25, t
        );
    }

    (function loop(t){requestAnimationFrame(loop);draw(t/1000);})(0);
})();

/* ═══════════════════════════════════════════════
   LAYER 4 — VOLUMETRIC LIGHT BEAMS
   Apple keynote / concert lighting style
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-beams');
    const cx=cv.getContext('2d');

    const BEAMS=[
        {x:.18, spread:0.04, h:42,  phase:0,    speed:.35},
        {x:.50, spread:0.06, h:48,  phase:1.05, speed:.28},
        {x:.82, spread:0.04, h:38,  phase:2.09, speed:.42},
        {x:.33, spread:0.03, h:200, phase:3.14, speed:.22},
        {x:.67, spread:0.03, h:160, phase:4.19, speed:.38},
    ];

    function draw(t){
        cx.clearRect(0,0,W,H);
        BEAMS.forEach(b=>{
            const sway=Math.sin(t*b.speed+b.phase)*0.04;
            const bx=(b.x+sway)*W;
            const topW=W*b.spread;
            const botW=W*(b.spread*4.5);
            const alpha=0.025+Math.sin(t*b.speed*1.3+b.phase)*0.015;
            const g=cx.createLinearGradient(bx,0,bx,H*0.7);
            g.addColorStop(0,   `hsla(${b.h},80%,70%,${alpha*2.5})`);
            g.addColorStop(0.3, `hsla(${b.h},70%,55%,${alpha})`);
            g.addColorStop(1,   'transparent');
            cx.beginPath();
            cx.moveTo(bx-topW,0);
            cx.lineTo(bx+topW,0);
            cx.lineTo(bx+botW,H*0.72);
            cx.lineTo(bx-botW,H*0.72);
            cx.closePath();
            cx.fillStyle=g;
            cx.fill();
        });
    }

    (function loop(t){requestAnimationFrame(loop);draw(t/1000);})(0);
})();

/* ═══════════════════════════════════════════════
   LAYER 5 — NEURAL PARTICLE MESH
   Gold nodes + connections, mouse-reactive
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-particles');
    const cx=cv.getContext('2d');
    let pts=[];

    function init(){
        const N=Math.floor(W*H/14000);
        pts=Array.from({length:N},()=>({
            x:Math.random()*W, y:Math.random()*H,
            vx:(Math.random()-.5)*.28, vy:(Math.random()-.5)*.28,
            r:Math.random()*1.6+.35,
            type:Math.random()<.72?0:(Math.random()<.55?1:2),
            ph:Math.random()*Math.PI*2,
            spd:.01+Math.random()*.02,
        }));
    }
    init();
    window.addEventListener('resize',init);

    const C=['rgba(201,168,76,','rgba(99,102,241,','rgba(16,185,129,'];
    const LINK=125;
    let frame=0;

    function draw(){
        requestAnimationFrame(draw);
        cx.clearRect(0,0,W,H);
        frame++;

        for(let i=0;i<pts.length;i++){
            const a=pts[i];
            for(let j=i+1;j<pts.length;j++){
                const b=pts[j];
                const dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy;
                if(d2<LINK*LINK){
                    const alpha=.1*(1-Math.sqrt(d2)/LINK);
                    cx.beginPath();
                    cx.strokeStyle=C[0]+alpha+')';
                    cx.lineWidth=.4;
                    cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke();
                }
            }
        }

        for(const p of pts){
            p.ph+=p.spd;
            const g=.45+Math.sin(p.ph)*.3;
            const col=C[p.type];
            const dx=p.x-MX,dy=p.y-MY,d2=dx*dx+dy*dy;
            if(d2<10000){
                const d=Math.sqrt(d2);
                const f=(100-d)/100*.5;
                p.vx+=dx/d*f;p.vy+=dy/d*f;
            }
            p.vx*=.992;p.vy*=.992;
            if(p.r>1.0){
                cx.beginPath();cx.arc(p.x,p.y,p.r*3.2,0,Math.PI*2);
                cx.fillStyle=col+(g*.045)+')';cx.fill();
            }
            cx.beginPath();cx.arc(p.x,p.y,p.r*(1+Math.sin(p.ph)*.18),0,Math.PI*2);
            cx.fillStyle=col+g+')';cx.fill();
            p.x+=p.vx;p.y+=p.vy;
            if(p.x<-20)p.x=W+20;if(p.x>W+20)p.x=-20;
            if(p.y<-20)p.y=H+20;if(p.y>H+20)p.y=-20;
        }

        if(frame%6===0){
            cx.beginPath();
            cx.strokeStyle='rgba(201,168,76,'+(Math.random()*.018+.003)+')';
            cx.lineWidth=.3;
            const y=Math.random()*H;
            cx.moveTo(0,y);cx.lineTo(W,y);cx.stroke();
        }
    }
    draw();
})();

/* ═══════════════════════════════════════════════
   LAYER 6 — FLOATING 3D DATA ORBS
   Depth-layered, parallax on mouse
═══════════════════════════════════════════════ */
(function(){
    const cv=document.getElementById('cv-orbs');
    const cx=cv.getContext('2d');

    const ORBS=Array.from({length:18},(_,i)=>({
        x:Math.random()*W, y:Math.random()*H,
        z:Math.random()*0.8+0.2,
        r:Math.random()*22+6,
        vx:(Math.random()-.5)*.18,
        vy:(Math.random()-.5)*.18,
        ph:Math.random()*Math.PI*2,
        spd:0.008+Math.random()*0.015,
        hue:Math.random()<.7?42:(Math.random()<.5?258:162),
        type:i%3,
    }));

    function draw(t){
        cx.clearRect(0,0,W,H);
        const px=(MX/W-.5)*30;
        const py=(MY/H-.5)*20;

        ORBS.sort((a,b)=>a.z-b.z).forEach(orb=>{
            orb.ph+=orb.spd;
            const bob=Math.sin(orb.ph)*4;
            const ox=orb.x+px*orb.z;
            const oy=orb.y+py*orb.z+bob;
            const r=orb.r*orb.z;
            const alpha=orb.z*0.55;

            /* Outer glow */
            const g1=cx.createRadialGradient(ox,oy,0,ox,oy,r*3.5);
            g1.addColorStop(0,`hsla(${orb.hue},80%,65%,${alpha*0.12})`);
            g1.addColorStop(1,'transparent');
            cx.beginPath();cx.arc(ox,oy,r*3.5,0,Math.PI*2);
            cx.fillStyle=g1;cx.fill();

            /* Core */
            const g2=cx.createRadialGradient(ox-r*.3,oy-r*.3,0,ox,oy,r);
            g2.addColorStop(0,`hsla(${orb.hue},90%,85%,${alpha*0.9})`);
            g2.addColorStop(0.5,`hsla(${orb.hue},75%,55%,${alpha*0.6})`);
            g2.addColorStop(1,`hsla(${orb.hue},60%,30%,${alpha*0.1})`);
            cx.beginPath();cx.arc(ox,oy,r,0,Math.PI*2);
            cx.fillStyle=g2;cx.fill();

            /* Ring */
            cx.beginPath();cx.arc(ox,oy,r,0,Math.PI*2);
            cx.strokeStyle=`hsla(${orb.hue},80%,70%,${alpha*0.4})`;
            cx.lineWidth=0.8;cx.stroke();

            orb.x+=orb.vx;orb.y+=orb.vy;
            if(orb.x<-50)orb.x=W+50;if(orb.x>W+50)orb.x=-50;
            if(orb.y<-50)orb.y=H+50;if(orb.y>H+50)orb.y=-50;
        });
    }

    (function loop(t){requestAnimationFrame(loop);draw(t/1000);})(0);
})();
</script></body></html>"""


def particle_background() -> str:
    return aurora_background()


def apex_motion_engine() -> str:
    """
    APEX Motion Engine v3 — injected into Streamlit parent document.
    Spring cursor · Scroll progress · Curtain entrance · Scroll reveals
    3D card tilt · Magnetic buttons · Number counters · Tab ripple
    Chart stagger · Parallax header · Hover micro-interactions
    """
    return r"""<!DOCTYPE html><html><head>
<style>html,body{margin:0;padding:0;height:1px;overflow:hidden;background:transparent;}</style>
</head><body><script>
(function(){
    let doc;
    try{doc=window.parent.document;}catch(e){return;}
    if(doc.__apexV3)return;
    doc.__apexV3=true;

    const S=doc.createElement('style');
    S.textContent=`
        *{cursor:none!important;}
        #_CD{
            position:fixed;width:6px;height:6px;background:#c9a84c;border-radius:50%;
            pointer-events:none;z-index:2147483647;left:0;top:0;
            transform:translate(-50%,-50%);mix-blend-mode:screen;will-change:left,top;
            transition:width .15s ease,height .15s ease,background .2s ease;
        }
        #_CD.clicking{width:3px;height:3px;background:#f5e080;}
        #_CR{
            position:fixed;width:32px;height:32px;
            border:1.5px solid rgba(201,168,76,.38);border-radius:50%;
            pointer-events:none;z-index:2147483646;left:0;top:0;
            transform:translate(-50%,-50%);will-change:left,top;
            transition:width .4s cubic-bezier(.34,1.56,.64,1),height .4s cubic-bezier(.34,1.56,.64,1),
                        border-color .3s ease,background .3s ease,border-radius .3s ease;
        }
        #_CR.hover{width:52px;height:52px;border-color:rgba(201,168,76,.8);background:rgba(201,168,76,.04);}
        #_CR.input{width:2px;height:20px;border-radius:2px;border-color:rgba(201,168,76,.9);background:rgba(201,168,76,.2);}
        #_CR.click{width:18px;height:18px;background:rgba(201,168,76,.2);border-color:rgba(201,168,76,1);}
        #_CR.drag{width:60px;height:60px;border-radius:8px;border-color:rgba(201,168,76,.6);background:rgba(201,168,76,.06);}
        #_SPB{
            position:fixed;top:0;left:0;height:2px;width:0%;
            background:linear-gradient(90deg,#c9a84c,#f5e080,#e8c96a,#c9a84c);
            background-size:200% auto;z-index:2147483645;pointer-events:none;
            box-shadow:0 0 10px rgba(201,168,76,.8),0 0 3px rgba(201,168,76,1);
            animation:_gs 2s linear infinite;transition:width .06s linear;
        }
        @keyframes _gs{0%{background-position:0% center}100%{background-position:200% center}}
        #_curtain{
            position:fixed;inset:0;
            background:linear-gradient(160deg,rgba(201,168,76,.08) 0%,rgba(2,2,8,.99) 100%);
            z-index:2147483640;pointer-events:none;transform-origin:top;
            animation:_curt .9s cubic-bezier(.76,0,.24,1) forwards;
        }
        @keyframes _curt{0%{transform:scaleY(1);opacity:1;}75%{transform:scaleY(1);opacity:1;}100%{transform:scaleY(0);opacity:0;}}
        ._rv{
            opacity:0!important;transform:translateY(28px) scale(.975)!important;filter:blur(5px)!important;
            transition:opacity .85s cubic-bezier(.22,1,.36,1),transform .85s cubic-bezier(.22,1,.36,1),filter .85s cubic-bezier(.22,1,.36,1)!important;
        }
        ._rv._vis{opacity:1!important;transform:translateY(0) scale(1)!important;filter:blur(0)!important;}
        .stButton>button:hover{box-shadow:0 8px 32px rgba(201,168,76,.28),0 0 0 1px rgba(201,168,76,.35)!important;}
        [data-baseweb="tab"]:hover{color:#c9a84c!important;background:rgba(201,168,76,.04)!important;}
        [data-testid="stMetric"]:hover{border-color:rgba(201,168,76,.4)!important;box-shadow:0 16px 50px rgba(0,0,0,.55),0 0 40px rgba(201,168,76,.08)!important;transform:translateY(-5px)!important;}
        .stDataFrame:hover{box-shadow:0 0 55px rgba(201,168,76,.07)!important;}
        .streamlit-expanderHeader:hover{border-color:rgba(201,168,76,.3)!important;color:#c9a84c!important;background:rgba(201,168,76,.04)!important;padding-left:1.3rem!important;transition:all .25s ease!important;}
        [data-testid="stChatMessage"]:hover{border-color:rgba(201,168,76,.16)!important;box-shadow:0 4px 20px rgba(0,0,0,.3)!important;transform:translateX(4px)!important;transition:all .25s ease!important;}
        [data-testid="stFileUploader"]:hover{border-color:rgba(201,168,76,.5)!important;box-shadow:0 0 55px rgba(201,168,76,.09),inset 0 0 55px rgba(201,168,76,.03)!important;}
        [data-testid="stDownloadButton"] button:hover{transform:translateY(-3px) scale(1.03)!important;box-shadow:0 10px 28px rgba(201,168,76,.22)!important;}
        .stSelectbox>div>div:hover{border-color:rgba(201,168,76,.35)!important;}
    `;
    doc.head.appendChild(S);

    const dot=doc.createElement('div');dot.id='_CD';
    const ring=doc.createElement('div');ring.id='_CR';
    const bar=doc.createElement('div');bar.id='_SPB';
    const curtain=doc.createElement('div');curtain.id='_curtain';
    doc.body.appendChild(dot);doc.body.appendChild(ring);
    doc.body.appendChild(bar);doc.body.appendChild(curtain);
    setTimeout(()=>{try{curtain.remove();}catch(e){}},1000);

    let mx=0,my=0,rx=0,ry=0,vx=0,vy=0;
    doc.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;dot.style.left=mx+'px';dot.style.top=my+'px';});
    (function loop(){const k=.12,d=.68;vx=(vx+(mx-rx)*k)*d;vy=(vy+(my-ry)*k)*d;rx+=vx;ry+=vy;ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(loop);})();

    doc.addEventListener('mouseover',e=>{
        const t=e.target;
        if(t.closest('input,textarea,select'))ring.className='input';
        else if(t.closest('[data-testid="stFileUploader"]'))ring.className='drag';
        else if(t.closest('button,[data-baseweb="tab"],.metric-card,[data-testid="stMetric"],.glass-card'))ring.className='hover';
        else ring.className='';
    });
    doc.addEventListener('mousedown',()=>{ring.classList.add('click');dot.classList.add('clicking');});
    doc.addEventListener('mouseup',()=>{ring.classList.remove('click');dot.classList.remove('clicking');});

    function updateBar(){const el=doc.documentElement;const p=el.scrollTop/(el.scrollHeight-el.clientHeight)*100||0;bar.style.width=Math.min(p,100)+'%';}
    doc.addEventListener('scroll',updateBar,{passive:true});
    const me=doc.querySelector('.main');if(me)me.addEventListener('scroll',updateBar,{passive:true});

    const io=new IntersectionObserver((entries)=>{
        entries.forEach((e,i)=>{if(e.isIntersecting){setTimeout(()=>e.target.classList.add('_vis'),i*50);io.unobserve(e.target);}});
    },{threshold:.04,rootMargin:'0px 0px -20px 0px'});

    function addReveal(){
        ['[data-testid="stMetric"]','[data-testid="stDataFrame"]','.js-plotly-plot',
         '[data-testid="stExpander"]','[data-testid="stDownloadButton"]',
         '.stMarkdown h3','.stMarkdown h2','.insight-box','.metric-card','.glass-card'
        ].forEach(sel=>doc.querySelectorAll(sel).forEach(el=>{if(!el._rv){el._rv=true;el.classList.add('_rv');io.observe(el);}}));
    }

    function addTilt(){
        doc.querySelectorAll('.metric-card,[data-testid="stMetric"],.glass-card').forEach(card=>{
            if(card._tilt)return;card._tilt=true;let fid;
            card.addEventListener('mousemove',e=>{
                cancelAnimationFrame(fid);fid=requestAnimationFrame(()=>{
                    const r=card.getBoundingClientRect();
                    const cx2=r.left+r.width/2,cy2=r.top+r.height/2;
                    const tx=((e.clientY-cy2)/(r.height/2))*-9;
                    const ty=((e.clientX-cx2)/(r.width/2))*9;
                    const dist=Math.sqrt((e.clientX-cx2)**2+(e.clientY-cy2)**2);
                    const glow=Math.max(0,(1-dist/(r.width*.6))*.28);
                    card.style.transform=`perspective(900px) rotateX(${tx}deg) rotateY(${ty}deg) scale(1.04) translateZ(6px)`;
                    card.style.boxShadow=`${-ty*1.4}px ${tx*1.4}px 40px rgba(0,0,0,.5),0 0 ${20+glow*60}px rgba(201,168,76,${glow}),inset 0 1px 0 rgba(201,168,76,.1)`;
                });
            });
            card.addEventListener('mouseleave',()=>{
                cancelAnimationFrame(fid);
                card.style.transition='transform .7s cubic-bezier(.34,1.56,.64,1),box-shadow .5s ease';
                card.style.transform='';card.style.boxShadow='';
                setTimeout(()=>card.style.transition='',800);
            });
        });
    }

    function addMagnetic(){
        doc.querySelectorAll('button').forEach(btn=>{
            if(btn._mag)return;btn._mag=true;let fid;
            btn.addEventListener('mousemove',e=>{
                cancelAnimationFrame(fid);fid=requestAnimationFrame(()=>{
                    const r=btn.getBoundingClientRect();
                    btn.style.transform=`translate(${(e.clientX-(r.left+r.width/2))*.25}px,${(e.clientY-(r.top+r.height/2))*.25}px)`;
                });
            });
            btn.addEventListener('mouseleave',()=>{
                cancelAnimationFrame(fid);
                btn.style.transition='transform .7s cubic-bezier(.34,1.56,.64,1)';
                btn.style.transform='';setTimeout(()=>btn.style.transition='',800);
            });
            btn.addEventListener('click',()=>{btn.style.transform='scale(.91)';setTimeout(()=>{btn.style.transform='';},160);});
        });
    }

    function addCounters(){
        doc.querySelectorAll('.metric-card h2,[data-testid="stMetricValue"]').forEach(el=>{
            if(el._cnt)return;el._cnt=true;
            const raw=el.textContent.replace(/[,% \t]/g,'').trim();
            const num=parseFloat(raw);
            if(isNaN(num)||num<=0||num>1e9)return;
            const hasPct=el.textContent.includes('%');
            const isInt=Number.isInteger(num)&&!el.textContent.includes('.');
            let start=null;const dur=1100;
            function easeOut(t){return 1-Math.pow(1-t,4);}
            function step(ts){
                if(!start)start=ts;const p=Math.min((ts-start)/dur,1);const v=num*easeOut(p);
                el.textContent=(isInt?Math.round(v).toLocaleString():v.toFixed(2))+(hasPct?'%':'');
                if(p<1)requestAnimationFrame(step);
                else el.textContent=(isInt?num.toLocaleString():num.toFixed(2))+(hasPct?'%':'');
            }
            setTimeout(()=>requestAnimationFrame(step),300);
        });
    }

    function addTabRipple(){
        doc.querySelectorAll('[data-baseweb="tab"]').forEach(tab=>{
            if(tab._rpl)return;tab._rpl=true;
            tab.addEventListener('click',e=>{
                const r=doc.createElement('span');const rect=tab.getBoundingClientRect();
                const size=Math.max(rect.width,rect.height);
                r.style.cssText=`position:absolute;width:${size}px;height:${size}px;border-radius:50%;background:rgba(201,168,76,.16);left:${e.clientX-rect.left-size/2}px;top:${e.clientY-rect.top-size/2}px;transform:scale(0);pointer-events:none;animation:_rpl .6s ease-out forwards;`;
                tab.style.position='relative';tab.style.overflow='hidden';tab.appendChild(r);setTimeout(()=>r.remove(),650);
            });
        });
        if(!doc.getElementById('_rplKF')){const ks=doc.createElement('style');ks.id='_rplKF';ks.textContent='@keyframes _rpl{to{transform:scale(3);opacity:0}}';doc.head.appendChild(ks);}
    }

    function addChartStagger(){
        doc.querySelectorAll('.js-plotly-plot').forEach((el,i)=>{
            if(el._cs)return;el._cs=true;
            el.style.opacity='0';el.style.transform='translateY(16px) scale(.985)';
            el.style.transition=`opacity .8s ease ${i*.07}s,transform .8s cubic-bezier(.22,1,.36,1) ${i*.07}s`;
            setTimeout(()=>{el.style.opacity='1';el.style.transform='translateY(0) scale(1)';},200+i*70);
        });
    }

    function addParallax(){
        // Disabled — causes header rotation on mouse move
    }

    function runAll(){addReveal();addTilt();addMagnetic();addCounters();addTabRipple();addChartStagger();addParallax();}
    runAll();
    new MutationObserver(()=>{clearTimeout(doc._apexT);doc._apexT=setTimeout(runAll,380);}).observe(doc.body,{childList:true,subtree:true});
})();
</script></body></html>"""


def apple_animations() -> str:
    return apex_motion_engine()
